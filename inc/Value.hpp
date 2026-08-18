#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

class Value : public std::enable_shared_from_this<Value> {
    private:
        double data; // holding the value of tensor
        double grad; // gradient

        std::vector<std::shared_ptr<void()>> prev; // vector of pointers towards parent nodes

        std::string op; // operand
        std::function<void()> backward; // backward function
    
    public:
        Value();
        ~Value();
};
