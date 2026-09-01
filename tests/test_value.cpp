#include <gtest/gtest.h>
#include <iostream>
#include "value.hpp"
#include <algorithm>

TEST(ADDITION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_ADDITION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(5);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    // Initialization verification
    EXPECT_EQ(a->get_grad(), 0);
    EXPECT_EQ(b->get_grad(), 0);
    EXPECT_EQ(a->get_prev().size(), 0);
    EXPECT_EQ(b->get_prev().size(), 0);
    
    // Value + Value
    std::shared_ptr<Value> c = a + b;
    c->backward();

    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(b->get_grad(), c->get_grad());
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), a) != c->get_prev().end(), true);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), true);
    EXPECT_EQ(c->get_prev().size(), 2);
    

    std::cout << "[PASSED] Case 1: Value + Value pointers addition has passed." << std::endl;

    // Value + double
    c = a + 2026;
    c->backward();

    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), 1);
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2026);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    std::cout << "[PASSED] Case 2: Value + double addition has passed." << std::endl;

    // double + Value
    c = 2027 + a;
    c->backward();

    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), 1);
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2027);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    std::cout << "[PASSED] Case 3: double + Value addition has passed." << std::endl;

}

TEST(SUBTRACTION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_SUBTRACTION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(5);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    EXPECT_EQ(a->get_grad(), 0);
    EXPECT_EQ(b->get_grad(), 0);


}

TEST(MULTIPLICATION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_MULTIPLICATION) {
    
}

TEST(DIVISION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_DIVISION) {
    
}