// Minimal dirent.h shim for Windows to provide opendir/readdir/closedir
#pragma once

#ifdef _WIN32
#include <windows.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <io.h>

#define MAX_DIRNAME 260

struct dirent {
    char d_name[MAX_DIRNAME];
};

typedef struct {
    HANDLE hFind;
    WIN32_FIND_DATAA data;
    struct dirent ent;
    int first;
} DIR;

static inline DIR *opendir(const char *name) {
    char pattern[MAX_DIRNAME];
    if (!name) return NULL;
    size_t len = strlen(name);
    if (len + 3 >= MAX_DIRNAME) return NULL;
    strcpy(pattern, name);
    // append wildcard
    if (pattern[len-1] == '/' || pattern[len-1] == '\\')
        strcat(pattern, "*");
    else {
        strcat(pattern, "\\*");
    }

    DIR *dir = (DIR*)malloc(sizeof(DIR));
    if (!dir) return NULL;
    dir->hFind = FindFirstFileA(pattern, &dir->data);
    if (dir->hFind == INVALID_HANDLE_VALUE) {
        free(dir);
        return NULL;
    }
    dir->first = 1;
    strncpy(dir->ent.d_name, dir->data.cFileName, MAX_DIRNAME-1);
    dir->ent.d_name[MAX_DIRNAME-1] = '\0';
    return dir;
}

static inline struct dirent *readdir(DIR *dir) {
    if (!dir) return NULL;
    if (dir->first) {
        dir->first = 0;
        strncpy(dir->ent.d_name, dir->data.cFileName, MAX_DIRNAME-1);
        dir->ent.d_name[MAX_DIRNAME-1] = '\0';
        return &dir->ent;
    }
    if (FindNextFileA(dir->hFind, &dir->data) == 0) return NULL;
    strncpy(dir->ent.d_name, dir->data.cFileName, MAX_DIRNAME-1);
    dir->ent.d_name[MAX_DIRNAME-1] = '\0';
    return &dir->ent;
}

static inline int closedir(DIR *dir) {
    if (!dir) return -1;
    int ok = 1;
    if (dir->hFind != INVALID_HANDLE_VALUE) ok = FindClose(dir->hFind);
    free(dir);
    return ok ? 0 : -1;
}

// Provide S_ISDIR/S_ISREG if missing
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & S_IFDIR) == S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & S_IFREG) == S_IFREG)
#endif

#endif /* _WIN32 */
