use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Data, DeriveInput, Fields, ItemFn, LitStr, Path, Type, parse_macro_input, spanned::Spanned,
};

#[proc_macro_attribute]
pub fn timer(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute argument (the message string)
    let message = parse_macro_input!(attr as Option<LitStr>);
    // Parse the function
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let fn_attrs = &input_fn.attrs;

    let message_str = message
        .and_then(|f| Some(f.value()))
        .unwrap_or(fn_sig.ident.to_string());

    // Generate the new function with timing code
    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            let __timer_start = std::time::Instant::now();
            log::info!("starting {}", #message_str);

            // Original function body wrapped to capture return value
            let __timer_result = (|| #fn_block)();

            log::info!("{}: finished in {:.2}s", #message_str, __timer_start.elapsed().as_secs_f32());

            __timer_result
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(CliValue, attributes(cli_value, cli))]
pub fn derive_cli_value(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let enum_ident = input.ident.clone();

    let Data::Enum(data_enum) = input.data else {
        return syn::Error::new(input.span(), "CliValue can only be derived for enums")
            .to_compile_error()
            .into();
    };

    let mut option_name = to_kebab_case(&enum_ident.to_string());
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
        let mut canonical = to_kebab_case(&variant_ident.to_string());
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

#[proc_macro_derive(CliOptions, attributes(cli))]
pub fn derive_cli_options(item: TokenStream) -> TokenStream {
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
        let parse_expr = build_cli_parse_expr(&field.ty, parse_with.as_ref());

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

#[proc_macro_derive(KvDisplay, attributes(kv))]
pub fn derive_kv_display(item: TokenStream) -> TokenStream {
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

    let format_parts: Vec<String> = keys
        .iter()
        .enumerate()
        .map(|(idx, key)| {
            let sep = if idx == 0 { "" } else { " " };
            format!("{sep}{}={{}}", key.value())
        })
        .collect();
    let format_lit = syn::LitStr::new(&format_parts.concat(), Span::call_site());

    let expanded = quote! {
        impl std::fmt::Display for #struct_ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, #format_lit, #(#vals),*)
            }
        }
    };

    TokenStream::from(expanded)
}

fn build_cli_parse_expr(ty: &Type, parse_with: Option<&Path>) -> proc_macro2::TokenStream {
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

fn to_kebab_case(s: &str) -> String {
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
