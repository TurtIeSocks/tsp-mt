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

        if let Some(seg) = path.segments.last() {
            if let PathArguments::AngleBracketed(AngleBracketedGenericArguments { args, .. }) =
                &seg.arguments
            {
                if let Some(GenericArgument::Type(t)) = args.first() {
                    return Some(t);
                }
            }
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
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(seg) = path.segments.last() {
            return format_ident!("{}", to_snake_case(&seg.ident.to_string()));
        }
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
