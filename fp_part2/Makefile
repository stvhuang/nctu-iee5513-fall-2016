CXX = nvcc
TARGET = cnnConvLayer.exe

all: cnnConvLayer.cu
	$(CXX) $< -o $(TARGET) -Wno-deprecated-gpu-targets

run: all
	./$(TARGET)

.PHONY: clean

clean:
	rm -f $(TARGET)
