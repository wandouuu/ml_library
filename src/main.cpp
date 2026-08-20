#include "value.hpp"
#include <iostream>

int main(){
    Value a = Value(5);
    std::cout << a.get_data();
    std::cout << a.get_grad();
    
    return 0;
}