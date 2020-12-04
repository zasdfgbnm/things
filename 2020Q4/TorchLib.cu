#include <utility>
#include <array>

struct ArgumentDef final {
  std::size_t i;
};

template <std::size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Is)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    std::array<ArgumentDef, sizeof...(Is)>{{ArgumentDef{Is}...}}
  );
}

int main() {
    constexpr auto returns = createArgumentVectorFromTypes(std::make_index_sequence<1>());
}
