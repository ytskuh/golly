// This file is part of Golly.
// See docs/License.html for the copyright notice.

#import "prefs.h"   // for SavePrefs

#import "PatternViewController.h"
#import "OpenViewController.h"
#import "SettingsViewController.h"
#import "HelpViewController.h"
#import "GollyAppDelegate.h"

// -----------------------------------------------------------------------------

// on iOS 18+ we need to force tab bar to bottom of screen

@interface BottomTabBarController : UITabBarController
@end

@implementation BottomTabBarController

- (void)viewDidLoad
{
    [super viewDidLoad];
    [self forceCompactTraitForBottomTabBar];
}

- (void)viewWillLayoutSubviews
{
    [super viewWillLayoutSubviews];
    // Re-apply on every layout pass — the system can reset this on rotation.
    [self forceCompactTraitForBottomTabBar];
}

- (void)forceCompactTraitForBottomTabBar
{
    if (@available(iOS 18.0, *)) {
        if (UIDevice.currentDevice.userInterfaceIdiom == UIUserInterfaceIdiomPad) {
            self.traitOverrides.horizontalSizeClass = UIUserInterfaceSizeClassCompact;
        }
    }
}

@end

// -----------------------------------------------------------------------------

@implementation GollyAppDelegate

@synthesize window = _window;

// -----------------------------------------------------------------------------

static UITabBarController *tabBarController = nil;      // for SwitchToPatternTab, etc

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    // set variable seed for later rand() calls
    srand((unsigned int)time(0));
    
    self.window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];

    UIViewController *vc0 = [[PatternViewController alloc] initWithNibName:nil bundle:nil];
    UIViewController *vc1 = [[OpenViewController alloc] initWithNibName:nil bundle:nil];
    UIViewController *vc2 = [[SettingsViewController alloc] initWithNibName:nil bundle:nil];
    UIViewController *vc3 = [[HelpViewController alloc] initWithNibName:nil bundle:nil];
    
    tabBarController = [[BottomTabBarController alloc] init];
    tabBarController.viewControllers = [NSArray arrayWithObjects:vc0, vc1, vc2, vc3, nil];
    
    self.window.rootViewController = tabBarController;
    
    [self.window makeKeyAndVisible];
    return YES;
}

// -----------------------------------------------------------------------------

- (void)applicationWillResignActive:(UIApplication *)application
{
    // this is called for certain types of temporary interruptions (such as an incoming phone call or SMS message)
    // or when the user quits the application and it begins the transition to the background state
    PauseGenTimer();
}

// -----------------------------------------------------------------------------

- (void)applicationDidEnterBackground:(UIApplication *)application
{
    // called when user hits home button
    PauseGenTimer();
    SavePrefs();
}

// -----------------------------------------------------------------------------

- (void)applicationWillEnterForeground:(UIApplication *)application
{
    // undo any changes made in applicationDidEnterBackground
    RestartGenTimer();
}

// -----------------------------------------------------------------------------

- (void)applicationDidBecomeActive:(UIApplication *)application
{
    // restart any tasks that were paused in applicationWillResignActive
    RestartGenTimer();
}

// -----------------------------------------------------------------------------

- (void)applicationWillTerminate:(UIApplication *)application
{
    // application is about to terminate
    // (never called in iOS 5, so use applicationDidEnterBackground)
}

@end // GollyAppDelegate

// =============================================================================

void SwitchToPatternTab()
{
    tabBarController.selectedIndex = 0;
}

// -----------------------------------------------------------------------------

void SwitchToOpenTab()
{
    tabBarController.selectedIndex = 1;
}

// -----------------------------------------------------------------------------

void SwitchToSettingsTab()
{
    tabBarController.selectedIndex = 2;
}

// -----------------------------------------------------------------------------

void SwitchToHelpTab()
{
    tabBarController.selectedIndex = 3;
}

// -----------------------------------------------------------------------------

UIViewController* CurrentViewController()
{
    return tabBarController.selectedViewController;
}

// -----------------------------------------------------------------------------

void EnableTabBar(bool enable)
{
    tabBarController.tabBar.userInteractionEnabled = enable;
}

// -----------------------------------------------------------------------------

void ShowTabBar(bool show)
{
    tabBarController.tabBar.hidden = !show;
}

// -----------------------------------------------------------------------------

CGFloat TabBarHeight()
{
    return tabBarController.tabBar.frame.size.height;
}

// -----------------------------------------------------------------------------

// this stuff replaces the deprecated UIActionSheet code that caused crashes on iOS 27

@interface PopoverMenuViewController ()
@property (nonatomic, copy) NSArray<NSString *> *titles;
@property (nonatomic, copy) void (^completion)(NSInteger);
@property (nonatomic, strong) NSArray<UIView *> *rows;   // buttons and separators, in display order
@property (nonatomic, assign) CGFloat rowHeight;
@property (nonatomic, assign) CGFloat separatorHeight;
@end

@implementation PopoverMenuViewController

- (instancetype)initWithTitles:(NSArray<NSString *> *)titles
                     completion:(void (^)(NSInteger))completion
{
    self = [super init];
    if (self) {
        _titles = [titles copy];
        _completion = [completion copy];

        UIFont *font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
        _rowHeight = MAX(44, ceil(font.lineHeight) + 16);
        _separatorHeight = 1.0 / [UIScreen mainScreen].scale;

        CGFloat totalHeight = (_rowHeight * titles.count) + (_separatorHeight * (titles.count - 1));
        self.preferredContentSize = CGSizeMake(240, totalHeight);
    }
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor systemBackgroundColor];

    UIFont *font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    NSMutableArray<UIView *> *rows = [NSMutableArray array];

    [self.titles enumerateObjectsUsingBlock:^(NSString *title, NSUInteger idx, BOOL *stop) {
        if (idx > 0) {
            UIView *separator = [[UIView alloc] init];
            separator.backgroundColor = [UIColor separatorColor];
            [self.view addSubview:separator];
            [rows addObject:separator];
        }

        UIButton *button = [UIButton buttonWithType:UIButtonTypeSystem];
        [button setTitle:title forState:UIControlStateNormal];
        button.titleLabel.font = font;
        button.tag = idx;
        [button addTarget:self action:@selector(buttonTapped:) forControlEvents:UIControlEventTouchUpInside];
        [self.view addSubview:button];
        [rows addObject:button];
    }];

    self.rows = rows;
}

- (void)viewDidLayoutSubviews
{
    [super viewDidLayoutSubviews];

    CGFloat width = self.view.bounds.size.width;
    CGFloat y = 0;

    for (UIView *row in self.rows) {
        CGFloat height = [row isKindOfClass:[UIButton class]] ? self.rowHeight : self.separatorHeight;
        row.frame = CGRectMake(0, y, width, height);
        y += height;
    }
}

- (void)buttonTapped:(UIButton *)sender
{
    NSInteger index = sender.tag;
    [self dismissViewControllerAnimated:YES completion:^{
        if (self.completion) self.completion(index);
    }];
}

@end // PopoverMenuViewController
