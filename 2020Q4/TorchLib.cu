#include <iostream>
#include <memory>
#include <type_traits>
#include <array>

template <typename T>
std::shared_ptr<bool> getTypePtr_() {
  return nullptr;
}

struct ArgumentDef final {
  using GetTypeFn = std::shared_ptr<bool>();
  GetTypeFn* getTypeFn;
};

template <typename... Ts, size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<Ts>}...}}
  );
}

int main() {
    constexpr auto returns = createArgumentVectorFromTypes<bool>(std::make_index_sequence<1>());
}
