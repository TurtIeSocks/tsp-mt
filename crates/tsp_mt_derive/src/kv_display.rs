use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input, spanned::Spanned};

pub fn derive_kv_display_inner(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let struct_ident = input.ident.clone();

    let Data::Struct(data_struct) = &input.data else {
        return syn::Error::new(input.span(), "KvDisplay can only be derived for structs")
            .to_compile_error()
            .into();
    };

    let Fields::Named(fields) = &data_struct.fields else {
        return syn::Error::new(input.span(), "KvDisplay requires named fields")
            .to_compile_error()
            .into();
    };

    let mut keys = Vec::new();
    let mut vals = Vec::new();

    for field in &fields.named {
        let Some(field_ident) = &field.ident else {
            continue;
        };
        let mut key = field_ident.to_string();
        let mut fmt_mode = String::from("display");

        for attr in &field.attrs {
            if !attr.path().is_ident("kv") {
                continue;
            }
            let parse_result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    let lit: LitStr = meta.value()?.parse()?;
                    key = lit.value();
                    return Ok(());
                }
                if meta.path.is_ident("fmt") {
                    let lit: LitStr = meta.value()?.parse()?;
                    fmt_mode = lit.value();
                    return Ok(());
                }
                Err(meta.error("unsupported kv attribute; expected name/fmt"))
            });
            if let Err(err) = parse_result {
                return err.to_compile_error().into();
            }
        }

        let key_lit = syn::LitStr::new(&key, Span::call_site());
        keys.push(key_lit);
        vals.push(match fmt_mode.as_str() {
            "display" => quote! { &self.#field_ident },
            "len" => quote! { &self.#field_ident.len() },
            "path" => quote! { &self.#field_ident.display() },
            other => {
                return syn::Error::new(field.span(), format!("unsupported kv fmt mode: {other}"))
                    .to_compile_error()
                    .into();
            }
        });
    }

    let longest = keys
        .iter()
        .max_by(|a, b| a.value().len().cmp(&b.value().len()))
        .map(|opt| opt.value().len())
        .unwrap_or(0);
    let format_parts: Vec<String> = keys
        .iter()
        .map(|key| {
            let value = key.value();
            let local_len = longest - value.len();
            let local_spaces = (0..local_len).map(|_| " ").collect::<String>();

            format!("\t{value}{local_spaces} = {{}}",)
        })
        .collect();
    let format_lit = syn::LitStr::new(&format!("\n{}", format_parts.join("\n")), Span::call_site());

    let expanded = quote! {
        impl std::fmt::Display for #struct_ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, #format_lit, #(#vals),*)
            }
        }
    };

    TokenStream::from(expanded)
}
