#include "rdc_runner/process.h"
#include "rdc_runner/path_utf8.h"

#include <cerrno>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cwctype>
#include <map>
#include <vector>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <vector>
#endif

namespace pvrgpu::rdc {
namespace {

#ifdef _WIN32

std::wstring Utf8ToWide(const std::string &value) {
  if (value.empty())
    return {};
  const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                       value.data(),
                                       static_cast<int>(value.size()), nullptr, 0);
  if (size <= 0)
    return {};
  std::wstring result(static_cast<std::size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), size);
  return result;
}

std::wstring QuoteWindowsArgument(const std::wstring &argument) {
  if (argument.empty())
    return L"\"\"";
  if (argument.find_first_of(L" \t\n\v\"") == std::wstring::npos)
    return argument;

  std::wstring quoted = L"\"";
  std::size_t backslashes = 0;
  for (wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
      continue;
    }
    if (character == L'\"') {
      quoted.append(backslashes * 2 + 1, L'\\');
      quoted.push_back(L'\"');
      backslashes = 0;
      continue;
    }
    quoted.append(backslashes, L'\\');
    backslashes = 0;
    quoted.push_back(character);
  }
  quoted.append(backslashes * 2, L'\\');
  quoted.push_back(L'\"');
  return quoted;
}

struct CaseInsensitiveWideLess {
  bool operator()(const std::wstring &left, const std::wstring &right) const {
    return std::lexicographical_compare(
        left.begin(), left.end(), right.begin(), right.end(),
        [](wchar_t a, wchar_t b) { return std::towlower(a) < std::towlower(b); });
  }
};

std::vector<wchar_t> BuildEnvironmentBlock(const ProcessRequest &request) {
  std::map<std::wstring, std::wstring, CaseInsensitiveWideLess> values;
  LPWCH block = GetEnvironmentStringsW();
  if (block) {
    for (const wchar_t *entry = block; *entry; entry += std::wcslen(entry) + 1) {
      const wchar_t *equals = std::wcschr(entry + (entry[0] == L'=' ? 1 : 0), L'=');
      if (equals)
        values.emplace(std::wstring(entry, equals), std::wstring(equals + 1));
    }
    FreeEnvironmentStringsW(block);
  }

  for (const std::string &name : request.unset_environment)
    values.erase(Utf8ToWide(name));
  for (const auto &[name, value] : request.environment)
    values[Utf8ToWide(name)] = Utf8ToWide(value);

  std::vector<wchar_t> result;
  for (const auto &[name, value] : values) {
    const std::wstring entry = name + L"=" + value;
    result.insert(result.end(), entry.begin(), entry.end());
    result.push_back(L'\0');
  }
  result.push_back(L'\0');
  return result;
}

std::string WindowsError(DWORD error) {
  wchar_t *message = nullptr;
  const DWORD size = FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, error, 0, reinterpret_cast<wchar_t *>(&message), 0, nullptr);
  std::string result = "Windows error " + std::to_string(error);
  if (size && message) {
    const int utf8_size = WideCharToMultiByte(CP_UTF8, 0, message,
                                               static_cast<int>(size), nullptr,
                                               0, nullptr, nullptr);
    if (utf8_size > 0) {
      result.assign(static_cast<std::size_t>(utf8_size), '\0');
      WideCharToMultiByte(CP_UTF8, 0, message, static_cast<int>(size),
                          result.data(), utf8_size, nullptr, nullptr);
      while (!result.empty() &&
             (result.back() == '\r' || result.back() == '\n'))
        result.pop_back();
    }
    LocalFree(message);
  }
  return result;
}

#endif

}  // namespace

ProcessResult RunProcess(const ProcessRequest &request) {
  ProcessResult result;
  std::error_code filesystem_error;
  std::filesystem::create_directories(request.stdout_path.parent_path(),
                                      filesystem_error);
  if (filesystem_error) {
    result.error = "cannot create stdout directory: " +
                   filesystem_error.message();
    return result;
  }
  std::filesystem::create_directories(request.stderr_path.parent_path(),
                                      filesystem_error);
  if (filesystem_error) {
    result.error = "cannot create stderr directory: " +
                   filesystem_error.message();
    return result;
  }

#ifdef _WIN32
  const std::wstring executable = request.executable.wstring();
  std::wstring command_line = QuoteWindowsArgument(executable);
  for (const std::string &argument : request.arguments) {
    command_line.push_back(L' ');
    command_line += QuoteWindowsArgument(Utf8ToWide(argument));
  }
  std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
  mutable_command.push_back(L'\0');

  SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE stdout_handle = CreateFileW(
      request.stdout_path.wstring().c_str(), GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE, &security, CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (stdout_handle == INVALID_HANDLE_VALUE) {
    result.error = "cannot open stdout log: " + WindowsError(GetLastError());
    return result;
  }
  HANDLE stderr_handle = CreateFileW(
      request.stderr_path.wstring().c_str(), GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE, &security, CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (stderr_handle == INVALID_HANDLE_VALUE) {
    result.error = "cannot open stderr log: " + WindowsError(GetLastError());
    CloseHandle(stdout_handle);
    return result;
  }

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = stdout_handle;
  startup.hStdError = stderr_handle;
  PROCESS_INFORMATION process{};
  std::vector<wchar_t> environment = BuildEnvironmentBlock(request);
  const BOOL created = CreateProcessW(
      executable.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
      CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
      environment.data(), nullptr, &startup, &process);
  CloseHandle(stdout_handle);
  CloseHandle(stderr_handle);
  if (!created) {
    result.error = "CreateProcess failed: " + WindowsError(GetLastError());
    return result;
  }
  result.started = true;

  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
                                sizeof(limits)))
      AssignProcessToJobObject(job, process.hProcess);
  }

  WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 1;
  if (!GetExitCodeProcess(process.hProcess, &exit_code))
    result.error = "GetExitCodeProcess failed: " + WindowsError(GetLastError());
  result.exit_code = static_cast<int>(exit_code);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
  if (job)
    CloseHandle(job);
  return result;
#else
  const int stdout_fd =
      open(request.stdout_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
  if (stdout_fd < 0) {
    result.error = "cannot open stdout log: " + std::string(std::strerror(errno));
    return result;
  }
  const int stderr_fd =
      open(request.stderr_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0666);
  if (stderr_fd < 0) {
    result.error = "cannot open stderr log: " + std::string(std::strerror(errno));
    close(stdout_fd);
    return result;
  }

  const pid_t child = fork();
  if (child < 0) {
    result.error = "fork failed: " + std::string(std::strerror(errno));
    close(stdout_fd);
    close(stderr_fd);
    return result;
  }
  if (child == 0) {
    if (dup2(stdout_fd, STDOUT_FILENO) < 0 ||
        dup2(stderr_fd, STDERR_FILENO) < 0)
      _exit(126);
    close(stdout_fd);
    close(stderr_fd);

    for (const std::string &name : request.unset_environment)
      unsetenv(name.c_str());
    for (const auto &[name, value] : request.environment)
      setenv(name.c_str(), value.c_str(), 1);

    std::vector<std::string> storage;
    storage.reserve(request.arguments.size() + 1);
    storage.push_back(PathToUtf8(request.executable));
    storage.insert(storage.end(), request.arguments.begin(),
                   request.arguments.end());
    std::vector<char *> argv;
    argv.reserve(storage.size() + 1);
    for (std::string &argument : storage)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    execv(request.executable.c_str(), argv.data());
    const std::string message = "exec failed: " +
                                std::string(std::strerror(errno)) + "\n";
    write(STDERR_FILENO, message.data(), message.size());
    _exit(errno == ENOENT ? 127 : 126);
  }

  result.started = true;
  close(stdout_fd);
  close(stderr_fd);
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno == EINTR)
      continue;
    result.error = "waitpid failed: " + std::string(std::strerror(errno));
    return result;
  }
  if (WIFEXITED(status))
    result.exit_code = WEXITSTATUS(status);
  else if (WIFSIGNALED(status))
    result.exit_code = 128 + WTERMSIG(status);
  else
    result.exit_code = 1;
  return result;
#endif
}

}  // namespace pvrgpu::rdc
