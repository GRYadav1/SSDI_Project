from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader
print("CHeckCOmping")
from sklearn_nn import train_model,test_Model
print("End Compile")

pathOrig ="E:\\Chrome-Downloads\\NewProject\\mysite\\polls\\data\\test\\050\\"
pathForg= "E:\\Chrome-Downloads\\NewProject\\mysite\\polls\\data\\test\\050_forg\\"
clf = None
def index(request):
    print("Before Call!!!")
    template = loader.get_template('polls/index.html')
    print("Hi index")
    global clf
    clf = train_model();
    return HttpResponse(template.render({}, request))

def imageSubmit(request):
    print("Hi imageSubmit")
    if 'q' in request.GET:
        pathString1=pathOrig+request.GET['p']
        pathString2= pathForg+request.GET['q']
        print(pathString1,pathString2);
        global clf
        message=test_Model(pathString1,pathString2,clf)
        print("Final Sore :",message)

        #pathString+=request.GET['q']
    else:
        message = 'You know , Jon Snow!'
        print(message)
    return HttpResponse(message)
