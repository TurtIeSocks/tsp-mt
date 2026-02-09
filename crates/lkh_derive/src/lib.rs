mod lkh_display;
mod with_methods;

use proc_macro::TokenStream;

#[proc_macro_derive(LkhDisplay, attributes(lkh))]
pub fn derive_lkh_display(input: TokenStream) -> TokenStream {
    lkh_display::derive_lkh_display_inner(input)
}

#[proc_macro_derive(WithMethods, attributes(with))]
pub fn derive_with_methods(input: TokenStream) -> TokenStream {
    with_methods::derive_with_methods_inner(input)
}
