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

fn parse_error_attr(input: &DeriveInput) -> syn::Result<Option<Type>> {
    let mut error_type = None;

    for attr in &input.attrs {
        if !attr.path().is_ident("with") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("error") {
                if error_type.is_some() {
                    return Err(meta.error("duplicate with(error = ...) attribute"));
                }
                let value = meta.value()?;
                error_type = Some(value.parse::<Type>()?);
                return Ok(());
            }
            Err(meta.error("unsupported with attribute on struct; expected error = <Type>"))
        })?;
    }

    Ok(error_type)
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
    let parsed_error_type = match parse_error_attr(&input) {
        Ok(value) => value,
        Err(err) => return err.to_compile_error().into(),
    };
    let default_error_ident = format_ident!("{}WithMethodsError", struct_ident);
    let error_type = parsed_error_type
        .map(|ty| quote!(#ty))
        .unwrap_or_else(|| quote!(#default_error_ident));

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
            let try_method_ident = format_ident!("try_with_{}", field_ident);
            methods.push(quote! {
                pub fn #method_ident<V>(mut self, #field_ident: V) -> Self
                where
                    V: ::core::convert::TryInto<#inner>,
                    <V as ::core::convert::TryInto<#inner>>::Error: ::core::fmt::Debug,
                {
                    self.#field_ident = Some(
                        #field_ident
                            .try_into()
                            .expect(concat!("failed to convert value for ", stringify!(#method_ident))),
                    );
                    self
                }
            });
            methods.push(quote! {
                pub fn #try_method_ident<V>(mut self, #field_ident: V) -> ::core::result::Result<Self, #error_type>
                where
                    V: ::core::convert::TryInto<#inner>,
                    <V as ::core::convert::TryInto<#inner>>::Error: ::core::fmt::Debug,
                    #error_type: ::core::convert::From<#default_error_ident>,
                {
                    if self.#field_ident.is_some() {
                        return Err(
                            <#error_type as ::core::convert::From<#default_error_ident>>::from(
                                #default_error_ident(stringify!(#field_ident)),
                            ),
                        );
                    }

                    self.#field_ident = Some(
                        #field_ident
                            .try_into()
                            .expect(concat!("failed to convert value for ", stringify!(#try_method_ident))),
                    );
                    Ok(self)
                }
            });
        } else {
            let ty = &field.ty;
            methods.push(quote! {
                pub fn #method_ident<V>(mut self, #field_ident: V) -> Self
                where
                    V: ::core::convert::TryInto<#ty>,
                    <V as ::core::convert::TryInto<#ty>>::Error: ::core::fmt::Debug,
                {
                    self.#field_ident = #field_ident
                        .try_into()
                        .expect(concat!("failed to convert value for ", stringify!(#method_ident)));
                    self
                }
            });
        }
    }

    let expanded = quote! {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct #default_error_ident(pub &'static str);

        impl ::core::fmt::Display for #default_error_ident {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                write!(f, "field already set: {}", self.0)
            }
        }

        impl ::std::error::Error for #default_error_ident {}

        impl #impl_generics #struct_ident #ty_generics #where_clause {
            #(#methods)*
        }
    };

    TokenStream::from(expanded)
}
