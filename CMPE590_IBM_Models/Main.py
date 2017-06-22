import TrainModels
import TestModels

while True:
    try:
        mode = int(input('\n\nPlease choose what you want to do: \n\t1: training \n\t2: testing \n\t3: give a sentence to translate\n\t9:for exit\n'))
    except ValueError:
        print ("Not a number")

    if mode == 1:
        TrainModels.train_models()
    elif mode == 2:
        try:
            test_model = int(input('Choose Model to test: \n\t1: IBM Model 1 \n\t2: IBM Model 2 \n\t3: IBM Model 3\n'))
        except ValueError:
            print ("Not a number")

        if test_model > 3 or test_model < 1:
            print("invalid number")
            exit()

        TestModels.test(test_model,False,'')
    elif mode == 3:
        sentence_to_translate = input("Plese provide sentence to translate: ")

        try:
            test_model = int(input('Choose Model to test: \n\t1: IBM Model 1 \n\t2: IBM Model 2 \n\t3: IBM Model 3\n'))
        except ValueError:
            print ("Not a number")

        TestModels.test(test_model,True,sentence_to_translate)
    elif mode == 9:
        break
    else:
        print("invalid mode")

print("goodbye!")