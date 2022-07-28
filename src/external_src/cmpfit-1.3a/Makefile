
OFILES=mpfit.o
LIBFILE=libmpfit.a

CFLAGS=-O3

all: $(LIBFILE) testmpfit

clean:
	rm -f $(OFILES) testmpfit $(LIBFILE)

mpfit.o: mpfit.c mpfit.h
	$(CC) -c -o $@ $< $(CFLAGS)

$(LIBFILE): $(OFILES)
	$(AR) r $@ $(OFILES)

testmpfit: testmpfit.c libmpfit.a
	$(CC) -o $@ $(CFLAGS) testmpfit.c -L. -lmpfit -lm
