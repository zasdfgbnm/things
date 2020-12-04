#include <iostream>
#include <memory>
#include <type_traits>
#include <array>


enum class TypeKind {
  BoolType
};

struct Type;
using TypePtr = std::shared_ptr<Type>;

struct Type : std::enable_shared_from_this<Type> {
private:
 TypeKind kind_;

protected:
 Type(TypeKind kind) : kind_(kind) {}
};


struct BoolType;
using BoolTypePtr = std::shared_ptr<BoolType>;
// This node represents a Python bool value
struct BoolType : public Type {
  static const TypeKind Kind = TypeKind::BoolType;
  // global singleton
  static BoolTypePtr get() {
    return BoolTypePtr(new BoolType());
  }

 private:
  BoolType() : Type(TypeKind::BoolType) {}
};

template <typename T>
struct getTypePtr_ final {
  static TypePtr call() {
    return BoolType::get();
  }
};

struct ArgumentDef final {
  using GetTypeFn = TypePtr();
  GetTypeFn* getTypeFn;
};

template <typename... Ts, size_t... Is>
constexpr std::array<ArgumentDef, sizeof...(Ts)> createArgumentVectorFromTypes(std::index_sequence<Is...>) {
  return (
    // Create the return value
    std::array<ArgumentDef, sizeof...(Ts)>{{ArgumentDef{&getTypePtr_<std::decay_t<Ts>>::call}...}}
  );
}

int main() {
    constexpr auto returns = createArgumentVectorFromTypes<bool>(std::make_index_sequence<1>());
}
