use quote::{format_ident, quote};
use syn::{
    AngleBracketedGenericArguments, GenericArgument, Ident, Path, PathArguments, Type, TypePath,
};

pub fn to_snake_case(name: &str) -> String {
    let mut snake_case = String::new();
    let name = name.replace(&['/', '\\', '.', '-'][..], "_");
    let mut prev_char = '\0'; // Track the previous character
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 && prev_char != '_' {
            snake_case.push('_');
        }
        snake_case.push(ch.to_ascii_lowercase());
        prev_char = ch; // Update the previous character
    }
    snake_case
}

// pub fn space_pascal_case(variant: &str) -> String {
//     let mut formatted = String::new();
//     let mut chars = variant.chars().peekable();

//     while let Some(c) = chars.next() {
//         if c.is_uppercase() && formatted.len() > 0 {
//             formatted.push(' ');
//         }
//         formatted.push(c);
//     }

//     formatted
// }

// pub fn to_pascal_case(input: &str) -> String {
//     input
//         .split('_')
//         .map(|word| {
//             let mut chars = word.chars();
//             match chars.next() {
//                 None => String::new(),
//                 Some(f) => f.to_uppercase().chain(chars).collect(),
//             }
//         })
//         .collect()
// }

// /// Helper function to convert a path to screaming snake case.
// pub(super) fn to_screaming_snake_case(path: &str) -> String {
//     to_snake_case(path).to_uppercase()
// }

pub fn inner_of_option(ty: &Type) -> Option<&Type> {
    if let Type::Path(TypePath { path, .. }) = ty {
        let is_supported_option_path = match path.segments.len() {
            1 => path.segments[0].ident == "Option",
            3 => {
                (path.segments[0].ident == "std" || path.segments[0].ident == "core")
                    && path.segments[1].ident == "option"
                    && path.segments[2].ident == "Option"
            }
            _ => false,
        };
        if !is_supported_option_path {
            return None;
        }

        if let Some(seg) = path.segments.last()
            && let PathArguments::AngleBracketed(AngleBracketedGenericArguments { args, .. }) =
                &seg.arguments
            && let Some(GenericArgument::Type(t)) = args.first()
        {
            return Some(t);
        }
    }
    None
}

pub fn is_phantom_data(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty {
        let is_supported_phantom_path = match path.segments.len() {
            1 => path.segments[0].ident == "PhantomData",
            3 => {
                (path.segments[0].ident == "std" || path.segments[0].ident == "core")
                    && path.segments[1].ident == "marker"
                    && path.segments[2].ident == "PhantomData"
            }
            _ => false,
        };
        if !is_supported_phantom_path {
            return false;
        }

        return path
            .segments
            .last()
            .is_some_and(|seg| matches!(seg.arguments, PathArguments::AngleBracketed(_)));
    }

    false
}

/// Try to derive a sensible snake-case identifier from a field’s type.
/// Falls back to `f{idx}` if we cannot get a usable path identifier.
pub fn ident_from_type(ty: &Type, idx: usize) -> Ident {
    if let Type::Path(TypePath { path, .. }) = ty
        && let Some(seg) = path.segments.last()
    {
        return format_ident!("{}", to_snake_case(&seg.ident.to_string()));
    }
    // Fallback
    format_ident!("f{idx}")
}

pub fn build_cli_parse_expr(ty: &Type, parse_with: Option<&Path>) -> proc_macro2::TokenStream {
    if let Some(parse_with) = parse_with {
        quote! { #parse_with(&raw)? }
    } else {
        quote! {
            raw.parse::<#ty>()
                .map_err(|e| crate::Error::invalid_input(format!(
                    "Invalid value for --{name}: {raw} ({e})"
                )))?
        }
    }
}

pub fn to_kebab_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for (idx, ch) in s.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if idx != 0 {
                out.push('-');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::{ident_from_type, inner_of_option, is_phantom_data, to_kebab_case, to_snake_case};

    #[test]
    fn to_snake_case_normalizes_separators_and_caps() {
        assert_eq!(to_snake_case("FooBar"), "foo_bar");
        assert_eq!(to_snake_case("a/b.c-d"), "a_b_c_d");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
    }

    #[test]
    fn inner_of_option_supports_short_std_and_core_paths() {
        let short_ty: syn::Type = parse_quote!(Option<String>);
        let std_ty: syn::Type = parse_quote!(std::option::Option<u8>);
        let core_ty: syn::Type = parse_quote!(core::option::Option<bool>);
        let non_opt: syn::Type = parse_quote!(Vec<String>);

        let short_inner = inner_of_option(&short_ty).expect("expected Option inner type");
        let std_inner = inner_of_option(&std_ty).expect("expected std Option inner type");
        let core_inner = inner_of_option(&core_ty).expect("expected core Option inner type");

        assert_eq!(quote::quote!(#short_inner).to_string(), "String");
        assert_eq!(quote::quote!(#std_inner).to_string(), "u8");
        assert_eq!(quote::quote!(#core_inner).to_string(), "bool");
        assert!(inner_of_option(&non_opt).is_none());
    }

    #[test]
    fn is_phantom_data_detects_supported_paths_with_generics() {
        let short_ty: syn::Type = parse_quote!(PhantomData<u8>);
        let std_ty: syn::Type = parse_quote!(std::marker::PhantomData<String>);
        let core_ty: syn::Type = parse_quote!(core::marker::PhantomData<bool>);
        let missing_generic: syn::Type = parse_quote!(PhantomData);
        let non_phantom: syn::Type = parse_quote!(Option<u8>);

        assert!(is_phantom_data(&short_ty));
        assert!(is_phantom_data(&std_ty));
        assert!(is_phantom_data(&core_ty));
        assert!(!is_phantom_data(&missing_generic));
        assert!(!is_phantom_data(&non_phantom));
    }

    #[test]
    fn ident_from_type_uses_last_path_segment_or_fallback() {
        let path_ty: syn::Type = parse_quote!(std::collections::HashMap<String, usize>);
        let tuple_ty: syn::Type = parse_quote!((u8, u8));

        assert_eq!(ident_from_type(&path_ty, 0).to_string(), "hash_map");
        assert_eq!(ident_from_type(&tuple_ty, 7).to_string(), "f7");
    }

    #[test]
    fn to_kebab_case_inserts_dashes_before_ascii_uppercase() {
        assert_eq!(to_kebab_case("SolverMode"), "solver-mode");
        assert_eq!(to_kebab_case("already-kebab"), "already-kebab");
        assert_eq!(to_kebab_case("X"), "x");
    }
}
