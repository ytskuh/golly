// This file is part of Golly.
// See docs/License.html for the copyright notice.

#import <UIKit/UIKit.h>

#import "InfoViewController.h"

// This view controller creates a modal dialog that appears
// when the user taps the Pattern tab's Save button.

@interface SaveViewController : UIViewController <UITextFieldDelegate, UITableViewDelegate, UITableViewDataSource>
{
    IBOutlet UITextField *nameText;
    IBOutlet UITableView *typeTable;
    IBOutlet UILabel *topLabel;
    IBOutlet UILabel *botLabel;
}

- (IBAction)doCancel:(id)sender;
- (IBAction)doSave:(id)sender;

@end

// Ask user to save given pattern or .rule file currently being edited.
void SaveTextFile(const char* filepath, const char* contents, InfoViewController* currentView);

// Ask user if they want to save their changes upon hitting the New button.
// The completion result is 0 if Cancel was tapped, 1 for Don't Save, or 2 for Save.
void SaveChanges(void (^completion)(int result));
