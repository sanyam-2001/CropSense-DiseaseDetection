import predictionFunctions


def predictPool(path):
    pathX = './files/' + path
    predictionModels = [predictionFunctions.predict_custom,
                        predictionFunctions.predict_Resnet,
                        predictionFunctions.predictMNv2_224,
                        predictionFunctions.predictMNv2_256]
    ans = []
    for model in predictionModels:
        ans.append(model(pathX))
    return ans
