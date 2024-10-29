import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.api.applications import VGG19
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, classification_report

# Tạo đường dẫn train test
Train_dir = "Dataset/train"
Test_dir = "Dataset/test"

# Tạo các biến thể của ảnh ( Làm giàu dữ liệu ảnh)
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

# Chuẩn hóa cho tập dữ liệu test
test_datagen = ImageDataGenerator(rescale=1./255)

# Load ảnh từ thư mục và áp dụng xử lý dữ liệu
train_generator = train_datagen.flow_from_directory(
    Train_dir,
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

# Xây mô hình VGG
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Đưa test vào theo dõi quá trình
test_losses = []
test_accuracies = []
class TestMetricsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = model.evaluate(test_generator, verbose=0)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f" - Test loss = {test_loss} - Test accuracy = {test_acc}")
# Huấn luyện mô hình mà không có tập validation
H = model.fit(
    train_generator,
    epochs=5,
    verbose=1,
    callbacks=[TestMetricsCallback()]
)

# Đánh giá mô hình qua test
score = model.evaluate(test_generator, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Vẽ biểu đồ chính xác và độ mất mát trong quá trình huấn luyện
fig = plt.figure()
numOfEpoch = 5
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), test_losses, label='test loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='training accuracy')
plt.plot(np.arange(0, numOfEpoch), test_accuracies, label='test accuracy')
plt.title('Accuracy and Loss - Training vs Test')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

# Ma trận nhầm lẫn và báo cáo
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Tạo ma trận nhầm lẫn (Confusion matrix)
cm = confusion_matrix(y_true, y_pred)
labels = ['healthy', 'tumor', 'stroke']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# In báo cáo phân loại (precision, recall, f1-score)
print(classification_report(y_true, y_pred, target_names=labels))

# Dự đoán trên một ảnh trong tập test
test_images, test_labels = next(test_generator)
plt.imshow(test_images[0])
y_predict = model.predict(test_images[0].reshape(1, 128, 128, 3))
print('Giá trị dự đoán: ', labels[np.argmax(y_predict)])
