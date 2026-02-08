use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, Path, parse_macro_input, spanned::Spanned};

use crate::utils;

pub fn derive_cli_options_inner(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let struct_ident = input.ident.clone();

    let Data::Struct(data_struct) = &input.data else {
        return syn::Error::new(input.span(), "CliOptions can only be derived for structs")
            .to_compile_error()
            .into();
    };

    let Fields::Named(fields) = &data_struct.fields else {
        return syn::Error::new(input.span(), "CliOptions requires named fields")
            .to_compile_error()
            .into();
    };

    let mut arms = Vec::new();

    for field in &fields.named {
        let Some(field_ident) = &field.ident else {
            continue;
        };

        let mut long_name: Option<String> = None;
        let mut parse_with: Option<Path> = None;

        for attr in &field.attrs {
            if !attr.path().is_ident("cli") {
                continue;
            }
            let parse_result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("long") {
                    let lit: LitStr = meta.value()?.parse()?;
                    long_name = Some(lit.value());
                    return Ok(());
                }
                if meta.path.is_ident("parse_with") {
                    let lit: LitStr = meta.value()?.parse()?;
                    parse_with = Some(syn::parse_str(&lit.value())?);
                    return Ok(());
                }
                Err(meta.error("unsupported cli attribute; expected long/parse_with"))
            });
            if let Err(err) = parse_result {
                return err.to_compile_error().into();
            }
        }

        let Some(long_name) = long_name else {
            continue;
        };
        let long_name_lit = syn::LitStr::new(&long_name, Span::call_site());
        let parse_expr = utils::build_cli_parse_expr(&field.ty, parse_with.as_ref());

        arms.push(quote! {
            #long_name_lit => {
                let raw = value.ok_or_else(|| {
                    crate::Error::invalid_input(format!("Missing value for --{name}"))
                })?;
                self.#field_ident = #parse_expr;
                Ok(true)
            }
        });
    }

    let expanded = quote! {
        impl #struct_ident {
            fn split_arg(
                raw_name: &str,
                args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
            ) -> (String, Option<String>) {
                if let Some((k, v)) = raw_name.split_once('=') {
                    return (k.to_string(), Some(v.to_string()));
                }

                let value = match args.peek() {
                    Some(next) if !next.starts_with("--") => args.next(),
                    _ => None,
                };

                (raw_name.to_string(), value)
            }

            fn apply_cli_option(
                &mut self,
                name: &str,
                value: Option<String>,
            ) -> crate::Result<bool> {
                match name {
                    #(#arms,)*
                    _ => Ok(false),
                }
            }
        }
    };

    TokenStream::from(expanded)
}
