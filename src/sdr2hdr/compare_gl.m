// A CAOpenGLLayer bridge. Rendering stays in libmpv's public render API.
// References: mpv/video/out/mac/gl_layer.swift; Apple's CAOpenGLLayer API.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>

typedef void (*DrawCallback)(int, int, int);

@interface SDRHDRComparisonLayer : CAOpenGLLayer {
@public
    CGLContextObj context;
    CGLPixelFormatObj format;
    DrawCallback callback;
    unsigned long frames;
}
@end

@implementation SDRHDRComparisonLayer
- (id)init {
    self = [super init];
    if (!self) return nil;
    CGLPixelFormatAttribute attributes[] = {
        kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core,
        kCGLPFAAccelerated, kCGLPFADoubleBuffer,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)64, kCGLPFAColorFloat, 0
    };
    GLint count;
    if (CGLChoosePixelFormat(attributes, &format, &count) != kCGLNoError || !format ||
        CGLCreateContext(format, NULL, &context) != kCGLNoError || !context) {
        [self release];
        return nil;
    }
    self.contentsFormat = kCAContentsFormatRGBA16Float;
    self.asynchronous = NO;
    self.opaque = YES;
    return self;
}
- (id)initWithLayer:(SDRHDRComparisonLayer *)other {
    self = [super initWithLayer:other];
    if (self) {
        context = CGLRetainContext(other->context);
        format = CGLRetainPixelFormat(other->format);
        callback = other->callback;
    }
    return self;
}
- (CGLPixelFormatObj)copyCGLPixelFormatForDisplayMask:(uint32_t)mask {
    return CGLRetainPixelFormat(format);
}
- (CGLContextObj)copyCGLContextForPixelFormat:(CGLPixelFormatObj)pixelFormat {
    return CGLRetainContext(context);
}
- (BOOL)canDrawInCGLContext:(CGLContextObj)ctx pixelFormat:(CGLPixelFormatObj)pf
             forLayerTime:(CFTimeInterval)t displayTime:(const CVTimeStamp *)ts {
    return callback != NULL;
}
- (void)drawInCGLContext:(CGLContextObj)ctx pixelFormat:(CGLPixelFormatObj)pf
          forLayerTime:(CFTimeInterval)t displayTime:(const CVTimeStamp *)ts {
    if (!callback) return;
    GLint viewport[4], fbo;
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
    callback(fbo, viewport[2], viewport[3]);
    glFlush();
    frames++;
}
- (void)dealloc {
    callback = NULL;
    if (context) CGLReleaseContext(context);
    if (format) CGLReleasePixelFormat(format);
    [super dealloc];
}
@end

void *comparison_layer_create(void) { return [[SDRHDRComparisonLayer alloc] init]; }
void comparison_layer_set_callback(SDRHDRComparisonLayer *layer, DrawCallback cb) { layer->callback = cb; }
void comparison_layer_make_current(SDRHDRComparisonLayer *layer) { CGLSetCurrentContext(layer->context); }
unsigned long comparison_layer_frames(SDRHDRComparisonLayer *layer) { return layer->frames; }
void comparison_layer_release(SDRHDRComparisonLayer *layer) { [layer release]; }

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_compare_gl", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__compare_gl(void) { return PyModule_Create(&module); }
