#include "value.hpp"
#include <iostream>

int main(){
    std::shared_ptr<Value> a = std::make_shared<Value>(5);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    std::shared_ptr<Value> c = a + b;

    std::cout << c->get_data() << std::endl;

    return 0;
}