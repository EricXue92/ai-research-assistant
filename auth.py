"""
auth.py — Email/password authentication via Supabase
"""

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def _get_client() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_ANON_KEY"],
    )


def is_logged_in() -> bool:
    return st.session_state.get("user") is not None


def show_auth_form() -> bool:
    """
    Render the login / sign-up form.
    Returns True if the user is authenticated, False otherwise.
    """
    if is_logged_in():
        return True

    supabase = _get_client()

    st.markdown("""
    <style>
    .auth-container { max-width: 420px; margin: 4rem auto; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📄 AI Research Assistant")
    st.caption("Upload research papers and ask questions — powered by Claude")
    st.divider()

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user = res.user
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tab_signup:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submitted:
            if not email or not password or not confirm:
                st.error("Please fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    if res.user:
                        st.success("Account created! You can now log in.")
                    else:
                        st.error("Sign up failed. Please try again.")
                except Exception as e:
                    st.error(f"Sign up failed: {e}")

    return False


def show_logout_button():
    """Render the signed-in user's email and a logout button in the sidebar."""
    user = st.session_state.get("user")
    if not user:
        return
    st.caption(f"Signed in as {user.email}")
    if st.button("Log Out", key="logout_btn"):
        try:
            _get_client().auth.sign_out()
        except Exception:
            pass
        st.session_state.user = None
        st.rerun()
