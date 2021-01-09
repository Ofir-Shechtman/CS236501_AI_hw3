import ID3
import KNN
import CostSensitiveKNN
import KNNForest
import improvedKNNForest


if __name__ == '__main__':
    print('__ID3__')
    ID3.main()
    #ID3.experiment()
    print('__KNN__')
    KNN.main()
    #KNN.experiment()
    print('__CostSensitiveKNN__')
    CostSensitiveKNN.main()
    print('__KNNForest__')
    KNNForest.main2()
    KNNForest.main()
    print('__improvedKNNForest__')
    improvedKNNForest.main2()
    improvedKNNForest.main()