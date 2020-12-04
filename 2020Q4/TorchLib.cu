#include <utility>
#include <array>

template <typename T>
bool getTypePtr_() {
  return false;
}

struct ArgumentDef final {
  using GetTypeFn = bool();
  GetTypeFn* getTypeFn;
};

template <typename... Ts, std::size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<Ts>}...}}
  );
}

int main() {
    constexpr auto returns = createArgumentVectorFromTypes<bool>(std::make_index_sequence<1>());
}
