/*
 * Simple Triangle with Color Shading Test Pattern.
 * Renders a single triangle with smooth per-vertex colors (Gouraud/color shading)
 * and captures the frame via RenderDoc API to produce simple_triangle.rdc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <renderdoc_app.h>

static const char *vertex_shader_source =
    "attribute vec2 a_position;\n"
    "attribute vec4 a_color;\n"
    "varying vec4 v_color;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_position, 0.0, 1.0);\n"
    "    v_color = a_color;\n"
    "}\n";

static const char *fragment_shader_source =
    "precision mediump float;\n"
    "varying vec4 v_color;\n"
    "void main() {\n"
    "    gl_FragColor = v_color;\n"
    "}\n";

typedef struct {
    float x, y;
    float r, g, b, a;
} Vertex;

static const Vertex triangle_vertices[3] = {
    { -0.6f, -0.6f,  1.0f, 0.0f, 0.0f, 1.0f }, /* Red */
    {  0.6f, -0.6f,  0.0f, 1.0f, 0.0f, 1.0f }, /* Green */
    {  0.0f,  0.6f,  0.0f, 0.0f, 1.0f, 1.0f }, /* Blue */
};

static GLuint compile_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char info[512];
        glGetShaderInfoLog(shader, sizeof(info), NULL, info);
        fprintf(stderr, "Shader compilation error: %s\n", info);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint create_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    if (!vs) return 0;
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glBindAttribLocation(program, 0, "a_position");
    glBindAttribLocation(program, 1, "a_color");
    glLinkProgram(program);

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char info[512];
        glGetProgramInfoLog(program, sizeof(info), NULL, info);
        fprintf(stderr, "Program link error: %s\n", info);
        glDeleteProgram(program);
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

static RENDERDOC_API_1_1_2 *get_renderdoc_api(void) {
    pRENDERDOC_GetAPI get_api = (pRENDERDOC_GetAPI)dlsym(RTLD_DEFAULT, "RENDERDOC_GetAPI");
    if (!get_api) {
        void *rdc_lib = dlopen("librenderdoc.dylib", RTLD_NOW | RTLD_GLOBAL);
        if (rdc_lib) {
            get_api = (pRENDERDOC_GetAPI)dlsym(rdc_lib, "RENDERDOC_GetAPI");
        }
    }
    if (!get_api) {
        fprintf(stderr, "RENDERDOC_GetAPI not found.\n");
        return NULL;
    }

    RENDERDOC_API_1_1_2 *rdc = NULL;
    int ret = get_api(eRENDERDOC_API_Version_1_1_2, (void **)&rdc);
    if (ret != 1 || !rdc) {
        fprintf(stderr, "Failed to initialize RenderDoc API.\n");
        return NULL;
    }
    return rdc;
}

int main(int argc, char **argv) {
    const char *capture_path = "simple_triangle";
    if (argc > 1) {
        capture_path = argv[1];
    } else {
        const char *env_path = getenv("RENDERDOC_CAPTURE_FILE");
        if (env_path && env_path[0]) {
            capture_path = env_path;
        }
    }

    RENDERDOC_API_1_1_2 *rdc = get_renderdoc_api();
    if (rdc) {
        printf("[simple_triangle] RenderDoc API initialized, capture template: %s\n", capture_path);
        rdc->SetCaptureFilePathTemplate(capture_path);
    } else {
        printf("[simple_triangle] RenderDoc API not available, running standalone.\n");
    }

    /* Initialize EGL */
    EGLDisplay display = EGL_NO_DISPLAY;
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (get_platform_display) {
        display = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
    }
    if (display == EGL_NO_DISPLAY) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, NULL, NULL)) {
        fprintf(stderr, "Failed to initialize EGL display\n");
        return 1;
    }

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "Failed to bind GLES API\n");
        return 1;
    }

    const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_configs = 0;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs < 1) {
        fprintf(stderr, "Failed to choose EGL config\n");
        return 1;
    }

    const int width = 512;
    const int height = 512;
    const EGLint pbuffer_attribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };

    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create pbuffer surface\n");
        return 1;
    }

    const EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context\n");
        return 1;
    }

    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "Failed to make EGL context current\n");
        return 1;
    }

    /* Start RenderDoc capture */
    if (rdc) {
        rdc->StartFrameCapture(NULL, NULL);
    }

    /* Compile program */
    GLuint program = create_program(vertex_shader_source, fragment_shader_source);
    if (!program) {
        fprintf(stderr, "Failed to create shader program\n");
        return 1;
    }
    glUseProgram(program);

    /* Viewport and clear */
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /* Setup vertex attributes */
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void *)&triangle_vertices[0].x);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void *)&triangle_vertices[0].r);

    /* Draw the color-shaded triangle */
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glFinish();

    /* End RenderDoc capture */
    if (rdc) {
        uint32_t success = rdc->EndFrameCapture(NULL, NULL);
        if (success) {
            printf("[simple_triangle] Frame capture succeeded!\n");
        } else {
            fprintf(stderr, "[simple_triangle] Frame capture failed.\n");
        }
    }

    eglSwapBuffers(display, surface);

    /* Cleanup */
    glDeleteProgram(program);
    eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(display, context);
    eglDestroySurface(display, surface);
    eglTerminate(display);

    printf("[simple_triangle] Render complete.\n");
    return 0;
}
