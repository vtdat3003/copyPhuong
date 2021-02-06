from tkinter import *
from PIL import Image, ImageTk
from tkinter import font
import cv2
from tkinter import filedialog
from plate_opencv import PlateModel
from detect_output import save_a_img

def result(output):
    x = ''
    for i in range(len(output)):
        x += output[i][1]
        if len(x) == 2:
            x += '-'
        if len(x) == 5:
            x += ' '
        if len(x) == 9:
            x += '.'
    return x


def Open_File():
    global display_area, imgtk, text
    filedir = filedialog.askopenfilename()
    model = PlateModel("./weights/plate-yolov3-tiny.cfg",
                       "./weights/plate-yolov3-tiny_last.weights",
                       "./weights/plate.names")
    try :
        if filedir != '':
            img = cv2.imread(filedir)
            img = cv2.resize(img, (300, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            display_area.create_image(0, 0, image=imgtk, anchor=NW)
            img_detect = save_a_img(filedir, "./test_detected/")
            x = model.predict(img_detect)
            predict_result = result(x)
            print(predict_result)
            text.set(predict_result)
    except:
        print("Fail")


windows = Tk()
windows.title("Number reader")
windows.geometry('800x300')
windows.resizable(width=False, height=False)
win_font = font.Font(size=20)
# -----------------------------------------
menubar = Frame(windows)
menubar.pack(fill=X)

filebt = Button(menubar, text='Open File', relief='flat', command=Open_File)
filebt.pack(side=LEFT)
# toolbt = Button(menubar, text='Tool',relief  = 'flat')
# toolbt.pack(side=LEFT)
separateline = Canvas(windows, width=810, height=1, bg='gray')
separateline.pack(side=TOP)
# ------------------------------------------
display_area = Canvas(windows, width=300, height=200, bg='white')
imgtk = None
display_area.pack(side=LEFT)

# ------------------------------------------
label = Label(windows, text='   Plate: ')
label['font'] = win_font
label.pack(side=LEFT)

# ------------------------------------------

text = StringVar()
text.set('')
entry = Entry(windows, textvariable=text)
entry['font'] = win_font
entry.pack(side=LEFT)

windows.mainloop()
