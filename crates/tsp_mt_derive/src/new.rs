use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Index, Type, parse_macro_input, spanned::Spanned};

use crate::utils;

pub fn derive_new_inner(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    /* -------- classify fields -------- */

    enum Kind<'a> {
        // record-style
        Named {
            ident: &'a Ident,
            ty: &'a Type,
        },
        OptionalNamed {
            ident: &'a Ident,
            inner: &'a Type,
        },
        Underscore(&'a Ident),

        // tuple-style
        Unnamed {
            _idx: usize,
            param: Ident,
            ty: &'a Type,
        },
        OptionalUnnamed {
            idx: usize,
            param: Ident,
            inner: &'a Type,
        },
    }

    let mut kinds = Vec::<Kind>::new();
    let mut used_param_idents = Vec::<Ident>::new(); // to avoid duplicates

    match &input.data {
        Data::Struct(data) => match &data.fields {
            /* ------ record struct ------ */
            Fields::Named(named) => {
                for field in &named.named {
                    let ident = field.ident.as_ref().unwrap();
                    let name_str = ident.to_string();

                    if name_str.starts_with('_') {
                        if !utils::is_phantom_data(&field.ty) {
                            return syn::Error::new(
                                field.span(),
                                "underscore-prefixed fields in `#[derive(New)]` must be `PhantomData<_>`",
                            )
                            .to_compile_error()
                            .into();
                        }
                        kinds.push(Kind::Underscore(ident));
                    } else if let Some(inner) = utils::inner_of_option(&field.ty) {
                        kinds.push(Kind::OptionalNamed { ident, inner });
                    } else {
                        kinds.push(Kind::Named {
                            ident,
                            ty: &field.ty,
                        });
                    }
                }
            }
            /* ------ tuple struct ------ */
            Fields::Unnamed(unnamed) => {
                for (idx, field) in unnamed.unnamed.iter().enumerate() {
                    // derive a parameter/builder ident
                    let mut param = utils::ident_from_type(&field.ty, idx);
                    // ensure uniqueness if two fields resolve to same snake-case
                    while used_param_idents.contains(&param) {
                        param = format_ident!("{}_{idx}", param);
                    }
                    used_param_idents.push(param.clone());

                    if let Some(inner) = utils::inner_of_option(&field.ty) {
                        kinds.push(Kind::OptionalUnnamed { idx, param, inner });
                    } else {
                        kinds.push(Kind::Unnamed {
                            _idx: idx,
                            param,
                            ty: &field.ty,
                        });
                    }
                }
            }
            Fields::Unit => {
                return syn::Error::new_spanned(&name, "`New` cannot be derived for unit structs")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&name, "`New` can only be derived for structs")
                .to_compile_error()
                .into();
        }
    }

    /* -------- build `new` signature & body -------- */

    let mut new_args = Vec::new(); // fn parameters
    let mut inits = Vec::new(); // struct ctor fields
    let mut builders = Vec::new(); // with_… methods

    for kind in &kinds {
        match kind {
            /* ------- record ------- */
            Kind::Named { ident, ty } => {
                new_args.push(quote! { #ident: #ty });
                inits.push(quote! { #ident });
                let method = format_ident!("with_{}", ident);
                builders.push(quote! {
                    pub fn #method(mut self, #ident: #ty) -> Self {
                        self.#ident = #ident;
                        self
                    }
                });
            }
            Kind::OptionalNamed { ident, inner } => {
                inits.push(quote! { #ident: None });
                let method = format_ident!("with_{}", ident);
                builders.push(quote! {
                    pub fn #method(mut self, #ident: #inner) -> Self {
                        self.#ident = Some(#ident);
                        self
                    }
                });
            }
            Kind::Underscore(ident) => {
                inits.push(quote! { #ident: std::marker::PhantomData });
            }

            /* ------- tuple ------- */
            Kind::Unnamed { param, ty, .. } => {
                new_args.push(quote! { #param: #ty });
                inits.push(quote! { #param });
            }
            Kind::OptionalUnnamed { idx, param, inner } => {
                let field_idx = Index::from(*idx);
                inits.push(quote! { None });
                let method = format_ident!("with_{}", param);
                builders.push(quote! {
                    pub fn #method(mut self, #param: #inner) -> Self {
                        self.#field_idx = Some(#param);
                        self
                    }
                });
            }
        }
    }

    let struct_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(_) => quote! { Self { #(#inits),* } },
            Fields::Unnamed(_) => quote! { Self ( #(#inits),* ) },
            Fields::Unit => unreachable!(),
        },
        _ => unreachable!(),
    };

    let expanded = quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Auto-generated constructor.
            pub const fn new(#(#new_args),*) -> Self {
                #struct_body
            }

            #(#builders)*
        }
    };

    TokenStream::from(expanded)
}
