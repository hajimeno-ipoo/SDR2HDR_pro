// libmpv renders into a Core Video texture shared with Metal.
// Apple: Mixing Metal and OpenGL rendering in a view.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <OpenGL/gl3.h>
#import <OpenGL/OpenGL.h>

typedef void (*DrawCallback)(int, int, int);

@interface SDRHDRComparisonLayer : CAMetalLayer {
@public
    CGLContextObj context;
    CGLPixelFormatObj format;
    DrawCallback callback;
    unsigned long frames;
    NSString *renderError;
@private
    CVPixelBufferRef pixels;
    CVOpenGLTextureCacheRef glCache;
    CVOpenGLTextureRef glTexture;
    CVMetalTextureCacheRef metalCache;
    CVMetalTextureRef metalTexture;
    GLuint fbo;
    id<MTLCommandQueue> commandQueue;
    int width, height;
}
- (void)releaseTextures;
- (int)renderFrame;
@end

@implementation SDRHDRComparisonLayer
- (id)init {
    self = [super init];
    if (!self) return nil;
    CGLPixelFormatAttribute attributes[] = {
        kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core,
        kCGLPFAAccelerated, 0
    };
    GLint count;
    if (CGLChoosePixelFormat(attributes, &format, &count) != kCGLNoError || !format ||
        CGLCreateContext(format, NULL, &context) != kCGLNoError || !context) {
        [self release];
        return nil;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    self.device = device;
    [device release];
    self.pixelFormat = MTLPixelFormatRGBA16Float;
    // Metal's blit encoder copies the shared image into the drawable.
    self.framebufferOnly = NO;
    self.opaque = YES;
    commandQueue = [self.device newCommandQueue];
    if (!commandQueue ||
        CVOpenGLTextureCacheCreate(NULL, NULL, context, format, NULL, &glCache) != kCVReturnSuccess ||
        CVMetalTextureCacheCreate(NULL, NULL, self.device, NULL, &metalCache) != kCVReturnSuccess) {
        [self release];
        return nil;
    }
    return self;
}

- (void)releaseTextures {
    if (fbo) glDeleteFramebuffers(1, &fbo);
    fbo = 0;
    if (glTexture) CFRelease(glTexture);
    glTexture = NULL;
    if (metalTexture) CFRelease(metalTexture);
    metalTexture = NULL;
    if (pixels) CFRelease(pixels);
    pixels = NULL;
    if (glCache) CVOpenGLTextureCacheFlush(glCache, 0);
    if (metalCache) CVMetalTextureCacheFlush(metalCache, 0);
    width = height = 0;
}

- (int)renderFrame {
    if (!callback || renderError) return renderError ? -1 : 0;
    int w = (int)(self.bounds.size.width * self.contentsScale);
    int h = (int)(self.bounds.size.height * self.contentsScale);
    if (w <= 0 || h <= 0) return 0;
    CGLSetCurrentContext(context);
    if (w != width || h != height) {
        [self releaseTextures];
        self.drawableSize = CGSizeMake(w, h);
        NSDictionary *attributes = @{
            (id)kCVPixelBufferOpenGLCompatibilityKey: @YES,
            (id)kCVPixelBufferMetalCompatibilityKey: @YES,
            (id)kCVPixelBufferIOSurfacePropertiesKey: @{}
        };
        if (CVPixelBufferCreate(NULL, w, h, kCVPixelFormatType_64RGBAHalf,
                               (CFDictionaryRef)attributes, &pixels) != kCVReturnSuccess ||
            CVOpenGLTextureCacheCreateTextureFromImage(NULL, glCache, pixels, NULL,
                                                       &glTexture) != kCVReturnSuccess ||
            CVMetalTextureCacheCreateTextureFromImage(NULL, metalCache, pixels, NULL,
                MTLPixelFormatRGBA16Float, w, h, 0, &metalTexture) != kCVReturnSuccess) {
            renderError = [@"OpenGLとMetalの共有画像を作成できません" retain];
            return -1;
        }
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            CVOpenGLTextureGetTarget(glTexture), CVOpenGLTextureGetName(glTexture), 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            renderError = [@"HDR画像の描画先を作成できません" retain];
            return -1;
        }
        width = w;
        height = h;
    }
    id<CAMetalDrawable> drawable = [self nextDrawable];
    if (!drawable) return 0;
    callback(fbo, width, height);
    // Finish OpenGL writes before Metal reads this shared image.
    glFinish();
    id<MTLCommandBuffer> command = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (!command || !blit) {
        renderError = [@"Metalの画像転送を開始できません" retain];
        return -1;
    }
    [blit copyFromTexture:CVMetalTextureGetTexture(metalTexture)
             sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(width, height, 1)
               toTexture:drawable.texture destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [command presentDrawable:drawable];
    [command commit];
    // Do not let the next OpenGL frame overwrite a texture still read by Metal.
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        renderError = [command.error.localizedDescription copy];
        return -1;
    }
    frames++;
    return 1;
}

- (void)dealloc {
    callback = NULL;
    CGLContextObj previous = CGLGetCurrentContext();
    if (context) CGLSetCurrentContext(context);
    [self releaseTextures];
    if (glCache) CFRelease(glCache);
    if (metalCache) CFRelease(metalCache);
    [commandQueue release];
    [renderError release];
    if (context) {
        CGLSetCurrentContext(previous == context ? NULL : previous);
        CGLReleaseContext(context);
    }
    if (format) CGLReleasePixelFormat(format);
    [super dealloc];
}
@end

void *comparison_layer_create(void) { return [[SDRHDRComparisonLayer alloc] init]; }
void comparison_layer_set_callback(SDRHDRComparisonLayer *layer, DrawCallback cb) { layer->callback = cb; }
void comparison_layer_make_current(SDRHDRComparisonLayer *layer) { CGLSetCurrentContext(layer->context); }
int comparison_layer_draw(SDRHDRComparisonLayer *layer) { return [layer renderFrame]; }
const char *comparison_layer_error(SDRHDRComparisonLayer *layer) { return layer->renderError.UTF8String; }
unsigned long comparison_layer_frames(SDRHDRComparisonLayer *layer) { return layer->frames; }
void comparison_layer_release(SDRHDRComparisonLayer *layer) { [layer release]; }

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_compare_gl", NULL, -1, NULL};
PyMODINIT_FUNC PyInit__compare_gl(void) { return PyModule_Create(&module); }
