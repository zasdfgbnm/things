#include <utility>
#include <type_traits>
#include <iostream>
#include <memory>

struct HasOperatorHelper
{
};

struct IsNullaryFunc
{
    template <typename T>
    static constexpr auto check(int)
        -> decltype((std::declval<T>()()), true)
    {
        return true;
    }

    template <typename T>
    static constexpr bool check(long)
    {
        return false;
    }
};

struct HasArrowOperator
{
    template <typename T>
    static constexpr auto check(int)
        -> decltype((std::declval<decltype(&T::operator->)>()), true)
    {
        return true;
    }

    template <typename T>
    static constexpr bool check(long)
    {
        return false;
    }
};

struct HasArrowStarOperator
{
    template <typename T>
    static constexpr auto check(int)
        -> decltype((std::declval<decltype(&T::operator->*)>()), true)
    {
        return true;
    }

    template <typename T>
    static constexpr bool check(long)
    {
        return false;
    }
};

struct TrueType
{
    static constexpr bool value()
    {
        return true;
    }
};

struct FalseType
{
    static constexpr bool value()
    {
        return false;
    }
};

template <typename T>
struct HasOperator
{
    constexpr operator HasOperatorHelper() const
    {
        return {};
    }

    template <typename T1>
    constexpr auto operator=(HasOperator<T1>) const
        -> decltype((std::declval<T>() = std::declval<T1>()), true)
    {
        return true;
    }

    constexpr bool operator=(HasOperatorHelper) const
    {
        return false;
    }

    template <typename... Ts>
    constexpr auto operator()(HasOperator<Ts>... args) const
        -> decltype((std::declval<T>()(std::declval<Ts>()...)), true)
    {
        return true;
    }

    constexpr bool operator()(HasOperatorHelper) const
    {
        return false;
    }

    template <typename T1 = int, std::enable_if_t<!IsNullaryFunc::check<T>(int{}), T1> = 0>
    constexpr bool operator()() const
    {
        return false;
    }

    template <typename... Ts>
    constexpr bool operator()(HasOperatorHelper, Ts... args) const
    {
        return false && operator()(args...);
    }

    template <typename T1>
    constexpr auto operator[](HasOperator<T1>) const
        -> decltype((std::declval<T>()[std::declval<T1>()]), true)
    {
        return true;
    }

    constexpr bool operator[](HasOperatorHelper) const
    {
        return false;
    }

    template <typename T1 = int, std::enable_if_t<HasArrowOperator::check<T>(int{}), T1> = 0>
    constexpr auto operator->() const -> TrueType *
    {
        return nullptr;
    }

    template <typename T1 = int, std::enable_if_t<!HasArrowOperator::check<T>(int{}), T1> = 0>
    constexpr auto operator->() const -> FalseType *
    {
        return nullptr;
    }

    template <typename T1, typename T2 = int, std::enable_if_t<HasArrowStarOperator::check<T>(int{}), T2> = 0>
    constexpr bool operator->*(T1) const
    {
        return true;
    }

    template <typename T1, typename T2 = int, std::enable_if_t<!HasArrowStarOperator::check<T>(int{}), T2> = 0>
    constexpr bool operator->*(T1) const
    {
        return false;
    }
};

#define DEFINE_UNARY_OP(op)                       \
    template <typename T1>                        \
    constexpr auto operator op(HasOperator<T1>)   \
        ->decltype(op std::declval<T1>(), true)   \
    {                                             \
        return true;                              \
    }                                             \
                                                  \
    constexpr bool operator op(HasOperatorHelper) \
    {                                             \
        return false;                             \
    }

#define DEFINE_UNARY_SUFFIX_OP(op)                     \
    template <typename T1>                             \
    constexpr auto operator op(HasOperator<T1>, int)   \
        ->decltype(std::declval<T1>() op, true)        \
    {                                                  \
        return true;                                   \
    }                                                  \
                                                       \
    constexpr bool operator op(HasOperatorHelper, int) \
    {                                                  \
        return false;                                  \
    }

#define DEFINE_BINARY_OP(op)                                         \
    template <typename T1, typename T2>                              \
    constexpr auto operator op(HasOperator<T1>, HasOperator<T2>)     \
        ->decltype((std::declval<T1>() op std::declval<T2>()), true) \
    {                                                                \
        return true;                                                 \
    }                                                                \
                                                                     \
    constexpr bool operator op(HasOperatorHelper, HasOperatorHelper) \
    {                                                                \
        return false;                                                \
    }

// Unary operators
DEFINE_UNARY_OP(+);
DEFINE_UNARY_OP(-);
DEFINE_UNARY_OP(~);
DEFINE_UNARY_OP(!);
DEFINE_UNARY_OP(++);
DEFINE_UNARY_OP(--);
DEFINE_UNARY_SUFFIX_OP(++);
DEFINE_UNARY_SUFFIX_OP(--);
DEFINE_UNARY_OP(*);
DEFINE_UNARY_OP(&);

// Binary operators
DEFINE_BINARY_OP(+);
DEFINE_BINARY_OP(-);
DEFINE_BINARY_OP(*);
DEFINE_BINARY_OP(/);
DEFINE_BINARY_OP(%);
DEFINE_BINARY_OP(&);
DEFINE_BINARY_OP(|);
DEFINE_BINARY_OP(^);
DEFINE_BINARY_OP(&&);
DEFINE_BINARY_OP(||);
DEFINE_BINARY_OP(<<);
DEFINE_BINARY_OP(>>);
DEFINE_BINARY_OP(==);
DEFINE_BINARY_OP(!=);
DEFINE_BINARY_OP(<);
DEFINE_BINARY_OP(>);
DEFINE_BINARY_OP(<=);
DEFINE_BINARY_OP(>=);

// Assignment operators
DEFINE_BINARY_OP(+=);
DEFINE_BINARY_OP(-=);
DEFINE_BINARY_OP(*=);
DEFINE_BINARY_OP(/=);
DEFINE_BINARY_OP(%=);
DEFINE_BINARY_OP(&=);
DEFINE_BINARY_OP(|=);
DEFINE_BINARY_OP(^=);
DEFINE_BINARY_OP(<<=);
DEFINE_BINARY_OP(>>=);

#undef DEFINE_UNARY_OP
#undef DEFINE_UNARY_SUFFIX_OP
#undef DEFINE_BINARY_OP

// comma operator
template <typename T1, typename T2>
constexpr auto operator,(HasOperator<T1>, HasOperator<T2>)
    -> decltype((std::declval<T1>(), std::declval<T2>()), true)
{
    return true;
}

constexpr bool operator,(HasOperatorHelper, HasOperatorHelper)
{
    return false;
}

// TODO: overload the following operators:
// <=> (requires C++20)
// ->* (requires pointer, unary and must be member)
// -> (requires pointer, unary and must be member)

template <typename T>
constexpr HasOperator<T> has_operator;

struct HasOperatorTestType
{
};

// Unary operators
static_assert(+has_operator<int>);
static_assert(!(+has_operator<HasOperatorTestType>));

static_assert(-has_operator<int>);
static_assert(!(-has_operator<HasOperatorTestType>));

static_assert(~has_operator<int>);
static_assert(!(~has_operator<HasOperatorTestType>));

static_assert(!has_operator<int>);
static_assert(!(!has_operator<HasOperatorTestType>));

static_assert(++has_operator<int &>);
static_assert(!(++has_operator<HasOperatorTestType &>));

static_assert(--has_operator<int &>);
static_assert(!(--has_operator<HasOperatorTestType &>));

static_assert(has_operator<int &> ++);
static_assert(!(has_operator<HasOperatorTestType &> ++));

static_assert(has_operator<int &> --);
static_assert(!(has_operator<HasOperatorTestType &> --));

// Comma
static_assert((has_operator<HasOperatorTestType>, has_operator<HasOperatorTestType>));
// TODO: how to test negative case for comma operator? I can not think of any case where comma operator is not valid.

// Binary operators

static_assert(has_operator<int> + has_operator<float>);
static_assert(!(has_operator<int> + has_operator<HasOperatorTestType>));

static_assert(has_operator<int> - has_operator<float>);
static_assert(!(has_operator<int> - has_operator<HasOperatorTestType>));

static_assert(has_operator<int> * has_operator<float>);
static_assert(!(has_operator<int> * has_operator<HasOperatorTestType>));

static_assert(has_operator<int> / has_operator<float>);
static_assert(!(has_operator<int> / has_operator<HasOperatorTestType>));

static_assert(has_operator<int> % has_operator<int>);
static_assert(!(has_operator<int> % has_operator<HasOperatorTestType>));

static_assert(has_operator<int> & has_operator<int>);
static_assert(!(has_operator<int> & has_operator<HasOperatorTestType>));

static_assert(has_operator<int> | has_operator<int>);
static_assert(!(has_operator<int> | has_operator<HasOperatorTestType>));

static_assert(has_operator<int> ^ has_operator<int>);
static_assert(!(has_operator<int> ^ has_operator<HasOperatorTestType>));

static_assert(has_operator<int> && has_operator<int>);
static_assert(!(has_operator<int> && has_operator<HasOperatorTestType>));

static_assert(has_operator<int> || has_operator<int>);
static_assert(!(has_operator<int> || has_operator<HasOperatorTestType>));

static_assert(has_operator<int> << has_operator<int>);
static_assert(!(has_operator<int> << has_operator<HasOperatorTestType>));

static_assert(has_operator<int> >> has_operator<int>);
static_assert(!(has_operator<int> >> has_operator<HasOperatorTestType>));

static_assert(has_operator<int> == has_operator<float>);
static_assert(!(has_operator<int> == has_operator<HasOperatorTestType>));

static_assert(has_operator<int> != has_operator<float>);
static_assert(!(has_operator<int> != has_operator<HasOperatorTestType>));

static_assert(has_operator<int> < has_operator<float>);
static_assert(!(has_operator<int> < has_operator<HasOperatorTestType>));

static_assert(has_operator<int> > has_operator<float>);
static_assert(!(has_operator<int> > has_operator<HasOperatorTestType>));

static_assert(has_operator<int> <= has_operator<float>);
static_assert(!(has_operator<int> <= has_operator<HasOperatorTestType>));

static_assert(has_operator<int> >= has_operator<float>);
static_assert(!(has_operator<int> >= has_operator<HasOperatorTestType>));

// Assignment operators
static_assert(has_operator<int &> = has_operator<int>);
static_assert(!(has_operator<int &> = has_operator<HasOperatorTestType>));

static_assert(has_operator<float &> += has_operator<int>);
static_assert(!(has_operator<int &> += has_operator<HasOperatorTestType>));

static_assert(has_operator<float &> -= has_operator<int>);
static_assert(!(has_operator<int &> -= has_operator<HasOperatorTestType>));

static_assert(has_operator<float &> *= has_operator<int>);
static_assert(!(has_operator<int &> *= has_operator<HasOperatorTestType>));

static_assert(has_operator<float &> /= has_operator<int>);
static_assert(!(has_operator<int &> /= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> %= has_operator<int>);
static_assert(!(has_operator<int &> %= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> &= has_operator<int>);
static_assert(!(has_operator<int &> &= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> |= has_operator<int>);
static_assert(!(has_operator<int &> |= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> ^= has_operator<int>);
static_assert(!(has_operator<int &> ^= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> <<= has_operator<int>);
static_assert(!(has_operator<int &> <<= has_operator<HasOperatorTestType>));

static_assert(has_operator<int &> >>= has_operator<int>);
static_assert(!(has_operator<int &> >>= has_operator<HasOperatorTestType>));

// Function call
int foo(int);
static_assert(has_operator<decltype(foo)>(has_operator<int>));
static_assert(!(has_operator<decltype(foo)>()));
static_assert(!(has_operator<decltype(foo)>(has_operator<HasOperatorTestType>)));
static_assert(!(has_operator<HasOperatorTestType>(has_operator<int>)));
int bar();
static_assert(has_operator<decltype(bar)>());
static_assert(!(has_operator<decltype(bar)>(has_operator<int>)));
static_assert(!(has_operator<decltype(bar)>(has_operator<HasOperatorTestType>)));
static_assert(!(has_operator<HasOperatorTestType>(has_operator<int>)));

// Array index
static_assert(has_operator<int[3]>[has_operator<int>]);
static_assert(!(has_operator<int[3]>[has_operator<HasOperatorTestType>]));
static_assert(!(has_operator<HasOperatorTestType>[has_operator<int>]));

// Arrow operator
static_assert(has_operator<std::unique_ptr<int>>->value());
static_assert(!has_operator<int>->value());

// Arrow star operator
struct OverloadArrowStar
{
    auto operator->*(int OverloadArrowStar::*memberPtr) const -> int *
    {
        return nullptr;
    }
};

static_assert(has_operator<OverloadArrowStar>->*2);
static_assert(has_operator<OverloadArrowStar>->*true);
static_assert(has_operator<OverloadArrowStar>->*has_operator<int>);
static_assert(!(has_operator<int>->*2));
static_assert(!(has_operator<int>->*true));
static_assert(!(has_operator<int>->*has_operator<OverloadArrowStar>));

int main()
{
    std::cout << (HasOperator<int>{} + HasOperator<int>{}) << std::endl;
    std::cout << (HasOperator<int>{} + HasOperator<double>{}) << std::endl;
    std::cout << (HasOperator<int>{} + HasOperator<std::ostream>{}) << std::endl;
}