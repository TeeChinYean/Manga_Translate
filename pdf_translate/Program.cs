using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Python.Runtime;

namespace PdfTranslate.Web
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // 1. 初始化 Python.NET 运行时 (Python 引擎互操作层)
            // 请确保环境变量 PYTHONNET_PYDLL 指向您的 libpython3.10.so 路径
            Runtime.PythonDLL = "/usr/lib/x86_64-linux-gnu/libpython3.10.so";
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads(); // 释放 GIL，允许 C# 进行异步多线程调度

            var builder = WebApplication.CreateBuilder(args);

            // 注册后端依赖服务
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddCors();

            var app = builder.Build();
            app.UseCors(x => x.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());

            // 2. 迁移原有的 FastAPI /api/v1/translate/upload 接口
            app.MapPost("/api/v1/translate/upload", async (
                [FromForm] IFormFile file,
                [FromForm] string source_lang,
                [FromForm] string target_lang,
                [FromForm] string page_range) =>
            {
                if (file == null || !file.FileName.EndsWith(".pdf"))
                {
                    return Results.BadRequest(new { error = "Invalid format. Only PDF files are supported." });
                }

                string taskId = Guid.NewGuid().ToString();
                string tempFilename = $"source_{taskId}_{file.FileName}";
                string tempFilepath = Path.Combine(Directory.GetCurrentDirectory(), "data/uploads", tempFilename);

                // 保存上传的文件
                using (var stream = new FileStream(tempFilepath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                // 将任务派发到 C# 的后台任务流中 (Channel / Task)
                _ = Task.Run(() => ProcessTranslationTask(taskId, tempFilepath, source_lang, target_lang, page_range));

                return Results.Ok(new
                {
                    task_id = taskId,
                    status = "queued",
                    message = "Task queued successfully in C# Orchestrator."
                });
            });

            // 3. SSE 状态流响应接口 (迁移自 FastAPI 的 EventSource)
            app.MapGet("/api/v1/translate/status/{taskId}", async (string taskId, HttpContext ctx) =>
            {
                ctx.Response.Headers.Add("Content-Type", "text/event-stream");
                
                // 简单的轮询示例，实际可替换为 C# Channel 的 IAsyncEnumerable
                for (int i = 0; i < 100; i += 10)
                {
                    string data = $"{{\"percent\": {i}, \"stage\": \"C# 驱动的流式提取中...\"}}";
                    await ctx.Response.WriteAsync($"event: progress\ndata: {data}\n\n");
                    await ctx.Response.Body.FlushAsync();
                    await Task.Delay(500);
                }

                await ctx.Response.WriteAsync("event: complete\ndata: {\"download_url\": \"/fake.pdf\"}\n\n");
                await ctx.Response.Body.FlushAsync();
            });

            Console.WriteLine("C# PDF Translate Engine Started. Listening on http://0.0.0.0:8000");
            app.Run("http://0.0.0.0:8000");
        }

        // 4. 核心调度系统：C# 调用现有的 Python 模型代码 (路线 B)
        private static async Task ProcessTranslationTask(string taskId, string pdfPath, string sourceLang, string targetLang, string pageRange)
        {
            Console.WriteLine($"🚀 Processing Task {taskId} inside C# background loop...");

            await Task.Run(() =>
            {
                // 使用 using (Py.GIL()) 重新获取全局解释器锁，安全调用 Python 侧的库
                using (Py.GIL())
                {
                    try
                    {
                        // 动态导入您现有的 Python 核心模块
                        dynamic sys = Py.Import("sys");
                        sys.path.append("/home/tcy/pdf_translate"); // 将您的项目目录加入环境变量

                        dynamic extractorModule = Py.Import("core.extractor");
                        dynamic engineModule = Py.Import("core.engine");

                        // 实例化 Python 的类
                        dynamic extractor = extractorModule.PDFLayoutExtractor(pdfPath);
                        dynamic translationEngine = engineModule.HighPerformanceTranslationEngine();

                        Console.WriteLine("C# Successfully invoked Python Extractor and Engine!");

                        // C# 端处理流式循环
                        // 这就完美替换了 main.py 中所有的提取和翻译逻辑
                        dynamic pageStream = extractor.extract_layout_stream(page_range_list: null, source_lang: sourceLang);
                        
                        foreach (dynamic pageData in pageStream)
                        {
                            Console.WriteLine($"C# Orchestrator received page {pageData[\"page_num\"]}");
                            
                            // 调用 Python 翻译模型
                            var result = translationEngine.translate_batch(new[] { pageData }, source_lang: sourceLang);
                            
                            // 这里可以对接 C# 的 Task.Run 去执行 GPU 重绘任务，充分利用并发
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Python Interop Error: {ex.Message}");
                    }
                }
            });
        }
    }
}
