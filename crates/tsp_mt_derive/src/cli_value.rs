use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input, spanned::Spanned};

use crate::utils;

pub fn derive_cli_value_inner(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let enum_ident = input.ident.clone();

    let Data::Enum(data_enum) = input.data else {
        return syn::Error::new(input.span(), "CliValue can only be derived for enums")
            .to_compile_error()
            .into();
    };

    let mut option_name = utils::to_kebab_case(&enum_ident.to_string());
    for attr in &input.attrs {
        if !attr.path().is_ident("cli_value") {
            continue;
        }
        let parse_result = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("option") {
                let lit: LitStr = meta.value()?.parse()?;
                option_name = lit.value();
                return Ok(());
            }
            Err(meta.error("unsupported cli_value attribute; expected option = \"...\""))
        });
        if let Err(err) = parse_result {
            return err.to_compile_error().into();
        }
    }

    let mut parse_arms = Vec::new();
    let mut display_arms = Vec::new();
    let mut expected_values = Vec::new();

    for variant in data_enum.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return syn::Error::new(
                variant.span(),
                "CliValue only supports enums with unit variants",
            )
            .to_compile_error()
            .into();
        }

        let variant_ident = variant.ident;
        let mut canonical = utils::to_kebab_case(&variant_ident.to_string());
        let mut aliases: Vec<String> = Vec::new();

        for attr in variant.attrs {
            if !attr.path().is_ident("cli") {
                continue;
            }
            let parse_result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let lit: LitStr = meta.value()?.parse()?;
                    canonical = lit.value();
                    return Ok(());
                }
                if meta.path.is_ident("alias") {
                    let lit: LitStr = meta.value()?.parse()?;
                    aliases.push(lit.value());
                    return Ok(());
                }
                Err(meta.error("unsupported cli attribute; expected name/alias"))
            });
            if let Err(err) = parse_result {
                return err.to_compile_error().into();
            }
        }

        expected_values.push(canonical.clone());
        let mut tokens = vec![syn::LitStr::new(&canonical, Span::call_site())];
        tokens.extend(
            aliases
                .iter()
                .map(|v| syn::LitStr::new(v, Span::call_site())),
        );
        let canonical_lit = syn::LitStr::new(&canonical, Span::call_site());

        parse_arms.push(quote! {
            #(#tokens)|* => Ok(Self::#variant_ident),
        });
        display_arms.push(quote! {
            Self::#variant_ident => #canonical_lit,
        });
    }

    let expected_joined = expected_values.join("|");
    let expected_lit = syn::LitStr::new(&expected_joined, Span::call_site());
    let option_lit = syn::LitStr::new(&option_name, Span::call_site());

    let expanded = quote! {
        impl #enum_ident {
            pub fn parse(raw: &str) -> crate::Result<Self> {
                match raw.to_ascii_lowercase().as_str() {
                    #(#parse_arms)*
                    _ => Err(crate::Error::invalid_input(format!(
                        "Invalid value for --{}: {} (expected {})",
                        #option_lit,
                        raw,
                        #expected_lit
                    ))),
                }
            }
        }

        impl std::fmt::Display for #enum_ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let value = match self {
                    #(#display_arms)*
                };
                write!(f, "{value}")
            }
        }
    };

    TokenStream::from(expanded)
}
