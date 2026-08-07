import bcrypt

from database.user_repository import get_user


def authenticate(
        username,
        password
):

    user = get_user(username)

    if user is None:
        return None

    if user["status"] != 1:
        return None

    if bcrypt.checkpw(
            password.encode(),
            user["password_hash"].encode()
    ):

        return dict(user)

    return None