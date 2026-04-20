#include <inferrt/model/IModel.h>

#include <iostream>

int main(int argc, char *argv[])
{
    auto model = std::unique_ptr<irt::model::IModel>(irt::model::CreateModel("alexnet"));
    return 0;
}