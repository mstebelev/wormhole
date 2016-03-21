include ../../make/ps_app.mk


clean:
	rm -rf build *.pb.*

ALIB=libwormholebase.a

$(ALIB): build/sample.pb.o
	ar cr $@ $+

all: $(ALIB)
