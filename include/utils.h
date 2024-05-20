#ifndef BGMNER_UTILS_H
#define BGMNER_UTILS_H

#include <Windows.h>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bgmner {
inline std::wstring s2ws(const std::string &str) {
    std::wstring wsTmp(str.begin(), str.end());
    return wsTmp;
}

inline std::string ws2s(const std::wstring &wstr) {
    std::string sTmp(wstr.begin(), wstr.end());
    return sTmp;
}

inline std::wstring utf8decode(const std::string &str) {
    // conv using Windows API
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    wchar_t *wstr = new wchar_t[len];
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, wstr, len);
    std::wstring ret = wstr;
    delete[] wstr;
    return ret;
}

inline std::string utf8encode(const std::wstring &wstr) {
    // conv using Windows API
    int len =
        WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    char *str = new char[len];
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, str, len, NULL, NULL);
    std::string ret = str;
    delete[] str;
    return ret;
}

inline std::string gbkenconde(const std::wstring &wstr) {
    // conv using Windows API
    int len =
        WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    char *str = new char[len];
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, str, len, NULL, NULL);
    std::string ret = str;
    delete[] str;
    return ret;
}
#include <stdio.h>

inline std::string outputChineseToConsole(const std::wstring &wstr) {
#ifdef _WIN32
    // conv using Windows API
    UINT cp = GetConsoleOutputCP();
    int len = WideCharToMultiByte(cp, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    char *str = new char[len];
    WideCharToMultiByte(cp, 0, wstr.c_str(), -1, str, len, NULL, NULL);
    std::string ret = str;
    delete[] str;
    return ret;
#else
    return utf8encode(wstr);
#endif
}

inline std::vector<std::wstring> splitByOneWChar(std::wstring str) {
    std::vector<std::wstring> ret;
    for (size_t i = 0; i < str.size(); i++) {
        ret.push_back(str.substr(i, 1));
    }
    return ret;
}

} // namespace bgmner.cpp

#endif // BGMNER_UTILS_H