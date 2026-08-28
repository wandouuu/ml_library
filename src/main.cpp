#include "value.hpp"
#include <iostream>

int main(){

    std::shared_ptr<Value> a = std::make_shared<Value>(5);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    std::shared_ptr<Value> c = 5 - b;

    std::cout << c->get_data() << std::endl;

    for(auto obj : c->get_prev()){
        std::cout << obj->get_data() << std::endl;
    }

    return 0;
}