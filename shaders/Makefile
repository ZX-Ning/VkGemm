SLANGC := slangc
SLANGFLAGS := -profile spirv_1_6 -capability spvCooperativeMatrixKHR -capability spvGroupNonUniform -O3

SOURCES := $(wildcard *.slang)
SPIRV := $(SOURCES:.slang=.spv)

.PHONY: all clean

all: $(SPIRV)

%.spv: %.slang
	$(SLANGC) $< $(SLANGFLAGS) -o $@

clean:
	rm -f $(SPIRV)
