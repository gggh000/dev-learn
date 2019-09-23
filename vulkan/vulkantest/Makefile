VULKAN_SDK_PATH = /root/vulkan/1.1.114.0/x86_64/include/

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

vulkanTutorial-phys-dev:
	g++ $(CFLAGS) -o vulkanTutorial-phys-dev vulkanTutorial-phys-dev.cpp $(LDFLAGS)

vulkanTutorial-validation-layer:  vulkanTutorial-validation-layer.cpp
	g++ $(CFLAGS) -o vulkanTutorial-validation-layer vulkanTutorial-validation-layer.cpp $(LDFLAGS)

vulkanTutorial-logical-dev:  vulkanTutorial-logical-dev.cpp
	g++ $(CFLAGS) -o vulkanTutorial-logical-dev vulkanTutorial-logical-dev.cpp $(LDFLAGS)

vulkanTutorial-window-surface: vulkanTutorial-window-surface.cpp
	g++ $(CFLAGS) -o vulkanTutorial-window-surface vulkanTutorial-window-surface.cpp $(LDFLAGS)

vulkanTutorial-swapchain: vulkanTutorial-swapchain.cpp
	g++ $(CFLAGS) -o vulkanTutorial-swapchain vulkanTutorial-swapchain.cpp $(LDFLAGS)
		
vulkanTutorial-imageview: vulkanTutorial-imageview.cpp
	g++ $(CFLAGS) -o vulkanTutorial-imageview vulkanTutorial-imageview.cpp $(LDFLAGS)

vulkanTutorial-pipeline: vulkanTutorial-pipeline.cpp
	g++ $(CFLAGS) -o vulkanTutorial-pipeline vulkanTutorial-pipeline.cpp $(LDFLAGS)

vulkanTutorial-pipelin-1: vulkanTutorial-pipelin-1.cpp
	g++ $(CFLAGS) -o vulkanTutorial-pipelin-1 vulkanTutorial-pipelin-1.cpp $(LDFLAGS)

vulkanTutorial-renderpass: vulkanTutorial-renderpass.cpp
	g++ $(CFLAGS) -o vulkanTutorial-renderpass vulkanTutorial-renderpass.cpp $(LDFLAGS)

vulkanTutorial-frameBuffer: vulkanTutorial-frameBuffer.cpp
	g++ $(CFLAGS) -o vulkanTutorial-frameBuffer vulkanTutorial-frameBuffer.cpp $(LDFLAGS)

vulkanTutorial-commandPool: vulkanTutorial-commandPool.cpp
	g++ $(CFLAGS) -o vulkanTutorial-commandPool  vulkanTutorial-commandPool.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest

