import os
import time

class LocalDocumentSkill:
    """
    Agent Skill: Word Processing & Document Generation
    This skill takes the extracted Japanese text and translated Chinese text,
    and generates a bilingual comparison document (.doc format using HTML tables)
    for professional translators to do post-editing (MTPE).
    
    Zero third-party dependencies required.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_bilingual_doc(self, task_id: str, pages_data: list, original_filename: str) -> str:
        """
        pages_data format:
        [
            {
                "page_num": 1,
                "blocks": [
                    {"raw": "こんにちは", "translated": "你好"}, ...
                ]
            }, ...
        ]
        """
        doc_path = os.path.join(self.output_dir, f"script_{task_id}.doc")
        
        # We use an HTML-based .doc format. Microsoft Word opens this natively and perfectly
        # formats it as a Word Document without needing python-docx dependency.
        html_content = [
            '<html><head><meta charset="utf-8">',
            '<style>',
            'body { font-family: "Microsoft YaHei", sans-serif; line-height: 1.6; padding: 20px; }',
            'h1 { text-align: center; color: #333; }',
            'table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }',
            'th, td { border: 1px solid #ccc; padding: 10px; text-align: left; vertical-align: top; }',
            'th { background-color: #f2f2f2; font-weight: bold; }',
            '.page-header { background-color: #e6f7ff; font-weight: bold; font-size: 1.1em; }',
            '.badge-ocr { display: inline-block; background: #e0f2fe; color: #0284c7; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-bottom: 4px; }',
            '.badge-llm { display: inline-block; background: #fef3c7; color: #d97706; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }',
            '</style>',
            f'<title>翻译台本 - {original_filename}</title>',
            '</head><body>',
            f'<h1>漫画翻译校对台本 (含模型调用追踪)</h1>',
            f'<p><strong>来源文件：</strong>{original_filename}</p>',
            f'<p><strong>生成时间：</strong>{time.strftime("%Y-%m-%d %H:%M:%S")}</p>',
            '<table>',
            '<tr><th width="10%">页码/气泡</th><th width="35%">原始日文 (OCR)</th><th width="35%">本地化中文翻译</th><th width="20%">所用模型与链路</th></tr>'
        ]
        
        for page in pages_data:
            p_num = page.get("page_num", "?")
            blocks = page.get("blocks", [])
            if not blocks:
                continue
                
            # Add page header row
            html_content.append(f'<tr class="page-header"><td colspan="4">第 {p_num} 页</td></tr>')
            
            for idx, blk in enumerate(blocks):
                raw = blk.get("raw", "").replace("\n", "<br>")
                trans = blk.get("translated", "").replace("\n", "<br>")
                ocr_eng = blk.get("ocr_engine", "MangaOCR (ViT)")
                trans_eng = blk.get("translation_engine", "Turbovec Qwen 3.5 4B")
                
                # Only show blocks that have actual text
                if not raw.strip():
                    continue
                    
                model_col = f'<span class="badge-ocr">OCR: {ocr_eng}</span><br><span class="badge-llm">LLM: {trans_eng}</span>'
                html_content.append(f'<tr><td>气泡 {idx+1}</td><td>{raw}</td><td>{trans}</td><td>{model_col}</td></tr>')
                
        html_content.append('</table></body></html>')
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
            
        return doc_path
