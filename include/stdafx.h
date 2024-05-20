#include <string>
#include <Windows.h>

namespace bgmner {

using char_t =
#ifdef UNICODE
    wchar_t;
    #define _T(x) L##x
#else
    char;
    #define _T(x) x
#endif



} // namespace bgmner.cpp

