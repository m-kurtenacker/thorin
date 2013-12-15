#ifdef THORIN_UF_TYPE
#define THORIN_U_TYPE(T) THORIN_UF_TYPE(T)
#define THORIN_F_TYPE(T) THORIN_UF_TYPE(T)
#endif

#ifndef THORIN_U_TYPE
#define THORIN_U_TYPE(T)
#endif

THORIN_U_TYPE(u1)
THORIN_U_TYPE(u8)
THORIN_U_TYPE(u16)
THORIN_U_TYPE(u32)
THORIN_U_TYPE(u64)

#ifndef THORIN_F_TYPE
#define THORIN_F_TYPE(T)
#endif

THORIN_F_TYPE(f32)
THORIN_F_TYPE(f64)

#undef THORIN_U_TYPE
#undef THORIN_F_TYPE
#undef THORIN_UF_TYPE
