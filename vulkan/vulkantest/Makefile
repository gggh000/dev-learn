VULKAN_SDK_PATH = /root/vulkan/1.1.114.0/x86_64/include/

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

vulkanTutorial-p41:  vulkanTutorial-p41.cpp
	g++ $(CFLAGS) -o vulkanTutorial-41 vulkanTutorial-p41.cpp $(LDFLAGS)
	
.PHONY: test clean

test: VulkanTest
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest

