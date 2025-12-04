function registerAuthenticationListener(dotNetHelper) {

    window.addEventListener('storage', (event) => {

        if (event.key === 'accessToken') {

            console.log('Access token was changed in another tab. Notifying Blazor...');

            dotNetHelper.invokeMethodAsync('OnTokenChanged');
        }
    });
}