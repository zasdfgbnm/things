#include <utility>
#include <iostream>
#include <typeinfo>

template <typename T, typename... Ts>
struct ForAllTypes
{
    template <typename Fun>
    void operator()(Fun f)
    {
        f((T *)nullptr);
        ForAllTypes<Ts...>{}(f);
    }
};

template <typename T>
struct ForAllTypes<T>
{
    template <typename Fun>
    void operator()(Fun f)
    {
        f((T *)nullptr);
    }
};

int main(int argc, char *argv[])
{
    auto f = [](auto *x)
    { using T = std::remove_pointer_t<decltype(x)>; std::cout << T(0.2) << std::endl; };
    ForAllTypes<bool, int, float>{}(f);
}
