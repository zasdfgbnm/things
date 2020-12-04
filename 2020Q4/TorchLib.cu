#include <iostream>
#include <memory>
#include <type_traits>
#include <array>

struct BoolType {};

template <typename T>
struct getTypePtr_ final {
  static std::shared_ptr<BoolType> call() {
    return nullptr;
  }
};

struct ArgumentDef final {
  using GetTypeFn = std::shared_ptr<BoolType>();
  GetTypeFn* getTypeFn;
};

template <typename... Ts, size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    // Create the return value
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<Ts>::call}...}}
  );
}

int main() {
    constexpr auto returns = createArgumentVectorFromTypes<bool>(std::make_index_sequence<1>());
}
