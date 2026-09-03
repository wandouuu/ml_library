#include <gtest/gtest.h>
#include <iostream>
#include "value.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

TEST(ADDITION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_ADDITION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(5.0);
    std::shared_ptr<Value> b = std::make_shared<Value>(3.0);

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

    // Value + double
    c = a + 2026;
    c->backward();

    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), 1);
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2026);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    // double + Value
    c = 2027 + a;
    c->backward();

    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), 1);
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2027);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

}

TEST(SUBTRACTION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_SUBTRACTION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(5.0);
    std::shared_ptr<Value> b = std::make_shared<Value>(3.0);

    // Initialization verification
    EXPECT_EQ(a->get_grad(), 0);
    EXPECT_EQ(b->get_grad(), 0);
    EXPECT_EQ(a->get_prev().size(), 0);
    EXPECT_EQ(b->get_prev().size(), 0);
    
    // Value - Value
    std::shared_ptr<Value> c = a - b;
    c->backward();

    EXPECT_EQ(c->get_data(), 2);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(b->get_grad(), -(c->get_grad()));
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), a) != c->get_prev().end(), true);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), true);
    EXPECT_EQ(c->get_prev().size(), 2);

    // Value - double
    c = a - 2026;
    c->backward();

    EXPECT_EQ(c->get_data(), -2021);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), -(c->get_grad()));
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2026);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    // double - Value
    c = 2027 - a;
    c->backward();
    
    EXPECT_EQ(c->get_data(), 2022);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), -(c->get_grad()));
    EXPECT_EQ(c->get_prev()[0]->get_grad(), c->get_grad());
    EXPECT_EQ(c->get_prev()[0]->get_data(), 2027);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

}

TEST(MULTIPLICATION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_MULTIPLICATION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(5);
    std::shared_ptr<Value> b = std::make_shared<Value>(3);

    // Initialization verification
    EXPECT_EQ(a->get_grad(), 0);
    EXPECT_EQ(b->get_grad(), 0);
    EXPECT_EQ(a->get_prev().size(), 0);
    EXPECT_EQ(b->get_prev().size(), 0);

    // Value * Value
    std::shared_ptr<Value> c = a * b;
    c->backward();

    EXPECT_EQ(c->get_data(), 15);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), b->get_data());
    EXPECT_EQ(b->get_grad(), a->get_data());
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), a) != c->get_prev().end(), true);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), true);
    EXPECT_EQ(c->get_prev().size(), 2);

    // Value * double
    c = a * 2026;
    c->backward();
    
    EXPECT_EQ(c->get_data(), 10130);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_prev()[1]->get_data());
    EXPECT_EQ(c->get_prev()[1]->get_grad(), a->get_data());
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2026);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    // double * Value
    c = 2027 * a;
    c->backward();
    
    EXPECT_EQ(c->get_data(), 10135);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_prev()[0]->get_data());
    EXPECT_EQ(c->get_prev()[0]->get_grad(), a->get_data());
    EXPECT_EQ(c->get_prev()[0]->get_data(), 2027);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

}

TEST(DIVISION_GRADIENT_TEST, TESTING_CALCULATIONS_GRADIENTS_DIVISION) {
    std::shared_ptr<Value> a = std::make_shared<Value>(1.0);
    std::shared_ptr<Value> b = std::make_shared<Value>(2.0);

    // Initialization verification
    EXPECT_EQ(a->get_grad(), 0);
    EXPECT_EQ(b->get_grad(), 0);
    EXPECT_EQ(a->get_prev().size(), 0);
    EXPECT_EQ(b->get_prev().size(), 0);

    // Value / Value
    std::shared_ptr<Value> c = a / b;
    c->backward();

    EXPECT_EQ(c->get_data(), 0.5);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad() * (1/b->get_data()));
    EXPECT_EQ(b->get_grad(), c->get_grad() * -(a->get_data() / std::pow(b->get_data(), 2)));
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), a) != c->get_prev().end(), true);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), true);
    EXPECT_EQ(c->get_prev().size(), 2);

    // Value / double
    c = a / 2026;
    c->backward();
    
    EXPECT_EQ(c->get_data(), a->get_data()/2026.0);
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(a->get_grad(), c->get_grad() * (1.0 / 2026.0));
    EXPECT_EQ(c->get_prev()[1]->get_grad(), c->get_grad() * - (a->get_data() / std::pow(2026, 2)));
    EXPECT_EQ(c->get_prev()[1]->get_data(), 2026);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), b) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    // double / Value
    c = 2027 / b;
    c->backward();
    
    EXPECT_EQ(c->get_data(), (2027.0 / 2.0));
    EXPECT_EQ(c->get_grad(), 1);
    EXPECT_EQ(b->get_grad(), c->get_grad() * -(c->get_prev()[0]->get_data() / std::pow(b->get_data(), 2)));
    EXPECT_EQ(c->get_prev()[0]->get_grad(), c->get_grad() / b->get_data());
    EXPECT_EQ(c->get_prev()[0]->get_data(), 2027);
    EXPECT_EQ(std::find(c->get_prev().begin(), c->get_prev().end(), a) != c->get_prev().end(), false);
    EXPECT_EQ(c->get_prev().size(), 2);

    // Edge cases
    // Division by 0 with non-zero numerator
    std::shared_ptr<Value> d = std::make_shared<Value>(2026.0);
    std::shared_ptr<Value> e = std::make_shared<Value>(0.0);

    std::shared_ptr<Value> f = d / e;
    f->backward();

    EXPECT_EQ(f->get_data(), std::numeric_limits<double>::infinity());
    EXPECT_EQ(d->get_grad(), std::numeric_limits<double>::infinity());
    EXPECT_EQ(e->get_grad(), -std::numeric_limits<double>::infinity());

    // Division by 0 with 0 numerator
    std::shared_ptr<Value> g = e / e;
    g->backward();
    
    EXPECT_EQ(std::isnan(g->get_data()), true);
    EXPECT_EQ(std::isnan(e->get_grad()), true);

}

TEST(TANH_TEST, TESTING_CALCULATIONS_TANH) {

    std::shared_ptr<Value> a = std::make_shared<Value>(5.0);

    std::shared_ptr<Value> b = tanh(a);
    b->backward();

    double tanh_eval = std::tanh(5.0);

    EXPECT_EQ(b->get_data(), tanh_eval);
    EXPECT_EQ(a->get_grad(), 1 - std::pow(tanh_eval, 2));
    EXPECT_EQ(b->get_prev()[0], a);
    EXPECT_EQ(b->get_prev().size(), 1);

}

TEST(RELU_TEST, TESTING_CALCULATIONS_RELU) {
    
}

TEST(SIGMOID_TEST, TESTING_CALCULATIONS_SIGMOID) {
    
}

TEST(BACKPROP_TEST, TESTING_CALCULATIONS_OVERALL_BACKPROP) {

}