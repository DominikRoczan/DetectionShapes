import cv2 as cv

img = cv.imread('image_test/Box_1.jpg')
# print(img.shape)

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)

# Kontury
contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, 0, (0, 0, 255), 2)
# print('1', contours[0])

# for i in contours[0]:
#     x,y = i[0]
#     print(x,y)

# Znalezienie prostokąta otaczającego pierwszy kontur
x, y, w, h = cv.boundingRect(contours[0])

# Wierzchołki
rect_vertices = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

# print(rect_vertices)
# print((center))

# Środek wszechświata
center = (int((x + w / 2)), int(y + h / 2))
cv.circle(img, center, 5, (255, 0, 0), 2)


# Prostoką w prostokącie-główny
# cv.rectangle(img, (p1_main), (p2_main), (255, 0, 0), 4)


def multi_bbox():
    ilosc_prostokatow = int(input('Wpisz ilość prostokątów: '))

    if ilosc_prostokatow == 0:
        print('Błąd: Podaj więcej niż 0!')
        return

    """
    Parametr szekosci bboxa. 
    Całkowita szerokość to lenght x 2.
    Całkowita szerokość bboxa to lenght w kierunku minusowym od środka, oraz leght  w kierunku plusowym od środka.
    Lengdt to 1/2 szerokości całkowitej bboxa.
    ...(-lenght) <--(0,0)--> (+lenght)...
    """
    lenght = 25

    # Szerokość/wysokość prostokąta od środka
    p1_main = int(x + w / 2 - lenght), y
    p2_main = int(x + w / 2 + lenght), y + h

    for i in range(0, ilosc_prostokatow):
        m = 5  # margines mdzy bboxami
        z = (i * (2 * lenght + 2 * m))  # margines *2 analogicznie do lenght

        px = (x + w / 2) - (x + w / 2 - z) # punkt przekroczenia

        if px > x:
            print('Przekroczenie bboxa!')
            return

        p3_l = (z - ((ilosc_prostokatow - 1) * (lenght + m)), 0)
        p4_l = (z - ((ilosc_prostokatow - 1) * (lenght + m)), 0)

        p3_left = tuple(x + y for x, y in zip(p1_main, p3_l))
        p4_left = tuple(x + y for x, y in zip(p2_main, p4_l))

        cv.rectangle(img, (p3_left), (p4_left), (0, 000, 250), 4)  # lewy

    cv.imshow('Obrazek', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


multi_bbox()
