import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.api.applications import VGG19
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add, BatchNormalization, Activation
from keras.api import Input, Model


# Tạo đường dẫn train val test
Train_dir = "Dataset/train"
Test_dir = "Dataset/test"

#Tạo các biến thể của ảnh ( Làm giàu dữ liệu ảnh)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Chuẩn hóa cho tập dữ liệu val và test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Load ảnh từ thư mục và a dụng xử lý dữ liệu
train_generator = train_datagen.flow_from_directory(
    Train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    Val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    Test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#Xây mô hình Resnet
def residual_block(x, filters, kernel_size=(3, 3), stride=(1, 1)):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if stride != (1, 1):
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Xây dựng mô hình ResNet50
input_shape = (128, 128, 3)
inputs = Input(shape=input_shape)

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 64)

x = residual_block(x, 128, stride=(2, 2))
x = residual_block(x, 128)
x = residual_block(x, 128)
x = residual_block(x, 128)

x = residual_block(x, 256, stride=(2, 2))
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)
x = residual_block(x, 256)

x = residual_block(x, 512, stride=(2, 2))
x = residual_block(x, 512)
x = residual_block(x, 512)

#các lớp fully phân loại
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

# Compile mô hình
model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Huấn luyện mô hình
H = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    verbose=1
)

#Đánh giá mô hình qua test
score = model.evaluate(test_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#vẽ biểu đồ chính xác và độ mất mát trong quá trình huấn luyện
fig = plt.figure()
numOfEpoch = 100
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

#Ma trận nhầm lẫn và báo cáo
test_images, test_labels = next(test_generator)
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

#Tạo ma trận nhầm lẫn (Confusion matrix)
cm = confusion_matrix(y_true, y_pred)
labels = ['healthy', 'tumor']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#In báo cáo phân loại (precision, recall, f1-score)
print(classification_report(y_true, y_pred))

# Dự đoán trên một ảnh trong tập test
test_images, test_labels = next(test_generator)
plt.imshow(test_images[0])
y_predict = model.predict(test_images[0].reshape(1, 128, 128, 3))
print('Giá trị dự đoán: ', labels[np.argmax(y_predict)])