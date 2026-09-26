using System;
using System.IO;
using System.Threading.Tasks;
using System.Threading.Channels;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Python.Runtime;
using System.Collections.Concurrent;
using System.Text.Json;

namespace PdfTranslate.CSharp
{
    public class Program
    {
        // 状态存储，模拟原本 main.py 中的 status_db
        private static readonly ConcurrentDictionary<string, object> StatusDb = new();

        public static void Main(string[] args)
        {
            Console.WriteLine("初始化 Python 引擎...");
            
            // 绑定系统全局的 Python 3.12 动态链接库
            Runtime.PythonDLL = "/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0";
            
            try
            {
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads(); // 必须释放 GIL，否则 C# 将被完全卡死
                Console.WriteLine("✅ Python 引擎初始化成功！");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Python 引擎初始化失败: {ex.Message}");
                Console.WriteLine("请确保系统中存在 libpython3.12.so.1.0，或在 setup 脚本中修正该路径。");
                return;
            }

            var builder = WebApplication.CreateBuilder(args);
            builder.Services.AddCors();
            
            // 提高 Kestrel 的上传文件大小限制
            builder.WebHost.ConfigureKestrel(serverOptions =>
            {
                serverOptions.Limits.MaxRequestBodySize = 524288000; // 500MB
            });

            // 必须同时提高 Form 绑定的限制 (默认 128MB)，否则上传大 PDF 会报 400 错误
            builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
            {
                options.ValueLengthLimit = int.MaxValue;
                options.MultipartBodyLengthLimit = 524288000; // 500MB
                options.MemoryBufferThreshold = int.MaxValue;
            });

            var app = builder.Build();
            app.UseCors(x => x.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());

            // 挂载静态文件目录 (对应 Python 的 app.mount("/static"))
            var staticPath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "static"));
            if (Directory.Exists(staticPath))
            {
                app.UseStaticFiles(new StaticFileOptions
                {
                    FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(staticPath),
                    RequestPath = "/static"
                });
            }

            // 根路由返回 index.html (对应 Python 的 @app.get("/"))
            app.MapGet("/", async ctx =>
            {
                string indexPath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "templates", "index.html"));
                if (File.Exists(indexPath))
                {
                    ctx.Response.ContentType = "text/html; charset=utf-8";
                    await ctx.Response.SendFileAsync(indexPath);
                }
                else
                {
                    ctx.Response.StatusCode = 404;
                    await ctx.Response.WriteAsync("Frontend index.html not found.");
                }
            });

            // 替代原有的 FastAPI Upload 接口
            app.MapPost("/api/v1/translate/upload", async (
                [FromForm] IFormFile? file,
                [FromForm] string? source_lang,
                [FromForm] string? target_lang,
                [FromForm] string? page_range) =>
            {
                source_lang ??= "ja";
                target_lang ??= "zh";
                page_range ??= "";

                if (file == null || !file.FileName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
                {
                    return Results.Json(new { error = "Invalid format. Only PDF files are supported." }, statusCode: 400);
                }

                string taskId = Guid.NewGuid().ToString();
                string tempDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "data", "uploads");
                Directory.CreateDirectory(tempDir);
                
                string tempFilepath = Path.Combine(tempDir, $"source_{taskId}_{file.FileName}");

                using (var stream = new FileStream(tempFilepath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                StatusDb[taskId] = new { percent = 0, stage = "C# 调度中心已接收任务", status = "queued" };

                // 使用 C# Task 异步触发后台大模型处理流
                _ = Task.Run(() => ProcessTranslationPipeline(taskId, tempFilepath, source_lang, target_lang, page_range));

                return Results.Ok(new
                {
                    task_id = taskId,
                    status = "queued",
                    message = "Task successfully dispatched by C# Orchestrator."
                });
            }).DisableAntiforgery();

            // 替代原有的 FastAPI SSE 接口
            app.MapGet("/api/v1/translate/status/{taskId}", async (string taskId, HttpContext ctx) =>
            {
                ctx.Response.Headers.Append("Content-Type", "text/event-stream");
                ctx.Response.Headers.Append("Cache-Control", "no-cache");
                ctx.Response.Headers.Append("Connection", "keep-alive");

                while (!ctx.RequestAborted.IsCancellationRequested)
                {
                    if (StatusDb.TryGetValue(taskId, out var statusObj))
                    {
                        var json = JsonSerializer.Serialize(statusObj);
                        string eventType = "progress";
                        if (json.Contains("\"status\":\"complete\"")) eventType = "complete";
                        else if (json.Contains("\"status\":\"error\"")) eventType = "error";

                        await ctx.Response.WriteAsync($"event: {eventType}\ndata: {json}\n\n");
                        await ctx.Response.Body.FlushAsync();
                        
                        // 简单的状态判断退出机制
                        if (eventType == "complete" || eventType == "error")
                        {
                            break;
                        }
                    }
                    await Task.Delay(500); // 500ms 轮询推送
                }
            });

            // 下载翻译后的 PDF (假设由 Python 引擎保存在同名 _translated.pdf 中)
            app.MapGet("/api/v1/download/{taskId}", async (string taskId, HttpContext ctx) =>
            {
                // TODO: 真正的逻辑需要去拿 Python 返回的文件路径，这里根据我们设定的临时目录来推断
                string tempDir = Path.Combine(Directory.GetCurrentDirectory(), "..", "data", "uploads");
                // 找到以 taskId 为后缀的翻译后文件
                var translatedFile = Directory.GetFiles(tempDir, $"source_{taskId}_*_translated.pdf").FirstOrDefault();
                if (translatedFile != null && File.Exists(translatedFile))
                {
                    // 设置好文件名供浏览器下载
                    var fileName = Path.GetFileName(translatedFile);
                    ctx.Response.Headers.Append("Content-Disposition", $"attachment; filename=\"{Uri.EscapeDataString(fileName)}\"");
                    await ctx.Response.SendFileAsync(translatedFile);
                    return;
                }
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsync("Translated PDF not found yet.");
            });

            Console.WriteLine("🚀 C# 工业级高并发汉化引擎启动成功 (http://0.0.0.0:8000)");
            app.Run("http://0.0.0.0:8000");
        }

        private static async Task ProcessTranslationPipeline(string taskId, string pdfPath, string sourceLang, string targetLang, string pageRange)
        {
            Console.WriteLine($"[Task {taskId}] 进入 C# 核心管道，准备唤醒 Python 大模型...");
            
            try
            {
                await Task.Run(() =>
                {
                    // 在调用任何 Python 代码前，C# 必须显式获取 GIL！
                    using (Py.GIL())
                    {
                        dynamic sys = Py.Import("sys");
                        // 确保能找到原本的 Python 模块
                        sys.path.append(Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..")));
                        
                        // 强制绑定依赖库路径 (Python.NET 嵌入模式默认不读取 ~/.local 和系统 dist-packages)
                        string venvPath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "venv", "lib", "python3.12", "site-packages"));
                        string homePath = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
                        string userLocalPath = string.IsNullOrEmpty(homePath) ? "" : Path.Combine(homePath, ".local", "lib", "python3.12", "site-packages");
                        
                        string[] systemPaths = new string[] {
                            "/usr/lib/python3/dist-packages",
                            "/usr/local/lib/python3.12/dist-packages",
                            "/usr/lib/python3.12/dist-packages",
                            "/usr/lib/python3.12/site-packages",
                            "/usr/local/lib/python3.12/site-packages"
                        };
                        foreach (var p in systemPaths) {
                            if (Directory.Exists(p)) sys.path.append(p);
                        }
                        
                        if (Directory.Exists(userLocalPath)) sys.path.insert(0, userLocalPath);
                        if (Directory.Exists(venvPath)) sys.path.insert(0, venvPath);
                        
                        // 导入我们的高并发 Python 流水线模块
                        dynamic pipelineModule = Py.Import("core.pipeline");
                        
                        // 构造状态回调，让 Python 流水线能实时把进度同步到 C# 的 StatusDb
                        Action<int, string, string> progressCallback = (percent, stage, status) =>
                        {
                            StatusDb[taskId] = new { 
                                percent = percent, 
                                stage = stage, 
                                status = status,
                                download_url = status == "complete" ? $"/api/v1/download/{taskId}" : null
                            };
                        };

                        // 唤醒高并发 Python 核心流水线
                        pipelineModule.run_pipeline(pdfPath, pageRange ?? "", sourceLang, targetLang, taskId, progressCallback);
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Task {taskId}] 执行崩溃: {ex}");
                StatusDb[taskId] = new { percent = 0, stage = $"系统崩溃: {ex.Message}", status = "error" };
            }
        }
    }
}
