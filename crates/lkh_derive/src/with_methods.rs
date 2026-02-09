use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Field, Fields, GenericArgument, PathArguments, Type, parse_macro_input,
    spanned::Spanned,
};

fn parse_skip_attr(field: &Field) -> syn::Result<bool> {
    let mut skip = false;

    for attr in &field.attrs {
        if !attr.path().is_ident("with") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                return Ok(());
            }
            Err(meta.error("unsupported with attribute; expected skip"))
        })?;
    }

    Ok(skip)
}

fn inner_of_option(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }

    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };

    if args.args.len() != 1 {
        return None;
    }

    let GenericArgument::Type(inner) = args.args.first()? else {
        return None;
    };

    Some(inner)
}

pub fn derive_with_methods_inner(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    let input_span = input.span();
    let struct_ident = input.ident.clone();
    let generics = input.generics.clone();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let data_struct = match input.data {
        Data::Struct(data) => data,
        _ => {
            return syn::Error::new(input_span, "WithMethods can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let fields = match data_struct.fields {
        Fields::Named(fields) => fields.named,
        _ => {
            return syn::Error::new(
                struct_ident.span(),
                "WithMethods requires a struct with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut methods = Vec::new();

    for field in fields {
        let Some(field_ident) = field.ident.clone() else {
            continue;
        };

        let skip = match parse_skip_attr(&field) {
            Ok(value) => value,
            Err(err) => return err.to_compile_error().into(),
        };
        if skip {
            continue;
        }

        let method_ident = format_ident!("with_{}", field_ident);

        if let Some(inner) = inner_of_option(&field.ty) {
            methods.push(quote! {
                pub fn #method_ident(mut self, #field_ident: impl Into<#inner>) -> Self {
                    self.#field_ident = Some(#field_ident.into());
                    self
                }
            });
        } else {
            let ty = &field.ty;
            methods.push(quote! {
                pub fn #method_ident(mut self, #field_ident: impl Into<#ty>) -> Self {
                    self.#field_ident = #field_ident.into();
                    self
                }
            });
        }
    }

    let expanded = quote! {
        impl #impl_generics #struct_ident #ty_generics #where_clause {
            #(#methods)*
        }
    };

    TokenStream::from(expanded)
}
