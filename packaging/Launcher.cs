using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("WatermarkRemover-AI")]
[assembly: AssemblyProduct("WatermarkRemover-AI")]
[assembly: AssemblyCompany("D-Ogi")]
[assembly: AssemblyVersion("__VERSION__.0")]
[assembly: AssemblyFileVersion("__VERSION__.0")]

internal static class Launcher
{
    [STAThread]
    private static int Main(string[] args)
    {
        bool check = args.Length == 1 && args[0] == "--check";
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string interpreter = Path.Combine(root, "python", check ? "python.exe" : "pythonw.exe");
        string script = Path.Combine(root, check ? "scripts/check_desktop.py" : "desktop_main.py");
        try
        {
            if (args.Length > 0 && !check) throw new ArgumentException("Supported argument: --check");
            if (!File.Exists(interpreter) || !File.Exists(script))
                throw new FileNotFoundException("Extract the complete portable archive before launching. Keep the EXE beside its python and ui folders.");
            ProcessStartInfo start = new ProcessStartInfo(interpreter, "\"" + script + "\"");
            start.WorkingDirectory = root;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            start.EnvironmentVariables.Remove("PYTHONHOME");
            start.EnvironmentVariables.Remove("PYTHONPATH");
            using (Process child = Process.Start(start))
            {
                child.WaitForExit();
                if (child.ExitCode != 0 && !check)
                    MessageBox.Show("The application could not start. Check data\\desktop.log in the extracted folder. Make sure the folder is writable.", "WatermarkRemover-AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            if (!check) MessageBox.Show(error.Message, "WatermarkRemover-AI", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
