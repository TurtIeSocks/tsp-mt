use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, LitStr, parse_macro_input, spanned::Spanned};

pub fn derive_as_lkh_inner(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let enum_ident = input.ident.clone();

    let Data::Enum(data_enum) = input.data else {
        return syn::Error::new(input.span(), "AsLkh can only be derived for enums")
            .to_compile_error()
            .into();
    };

    let mut arms = Vec::new();

    for variant in data_enum.variants {
        let variant_ident = variant.ident.clone();

        let mut mappings = Vec::new();
        for attr in &variant.attrs {
            if !attr.path().is_ident("lkh") {
                continue;
            }

            if let Ok(lit) = attr.parse_args::<LitStr>() {
                mappings.push((None, lit));
                continue;
            }

            let mut value: Option<LitStr> = None;
            let mut pat: Option<LitStr> = None;
            let parse_result = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("value") {
                    value = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                if meta.path.is_ident("pat") {
                    pat = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                Err(meta.error("unsupported lkh attribute; expected \"...\" or value/pat"))
            });
            if let Err(err) = parse_result {
                return err.to_compile_error().into();
            }

            let Some(value) = value else {
                return syn::Error::new(attr.span(), "missing lkh value")
                    .to_compile_error()
                    .into();
            };
            mappings.push((pat, value));
        }

        if mappings.is_empty() {
            return syn::Error::new(
                variant.span(),
                "missing #[lkh(...)] mapping for enum variant",
            )
            .to_compile_error()
            .into();
        }

        for (pat_lit, value_lit) in mappings {
            let pattern = match (&variant.fields, pat_lit) {
                (Fields::Unit, None) => quote! { Self::#variant_ident },
                (Fields::Unit, Some(_)) => {
                    return syn::Error::new(
                        variant.span(),
                        "unit variants cannot use lkh pat; use #[lkh(\"...\")] or value = \"...\"",
                    )
                    .to_compile_error()
                    .into();
                }
                (_, Some(pat_lit)) => {
                    let parsed_pat =
                        match syn::parse_str::<proc_macro2::TokenStream>(&pat_lit.value()) {
                            Ok(pat) => pat,
                            Err(err) => return err.to_compile_error().into(),
                        };
                    quote! { Self::#variant_ident #parsed_pat }
                }
                (_, None) => {
                    return syn::Error::new(
                        variant.span(),
                        "non-unit variants require lkh pat, e.g. #[lkh(value = \"...\", pat = \"{ field: value }\")]",
                    )
                    .to_compile_error()
                    .into();
                }
            };

            arms.push(quote! {
                #pattern => #value_lit,
            });
        }
    }

    let expanded = quote! {
        impl #enum_ident {
            fn as_lkh(self) -> &'static str {
                match self {
                    #(#arms)*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
